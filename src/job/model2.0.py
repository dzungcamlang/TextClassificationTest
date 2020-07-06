'''
Created on 2020年6月2日

@author: Administrator
'''
#一个端到端的模型，用于预测答案的span。

import sys, os, random
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time, environment, config_loaders
import pickle
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from job.data_preprocess import DataProcessor
from job import data_preprocess
import time
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"
class Model():
    
    def __init__(self, mode="work"):
        self.mode = mode
        self.clip = 5.0
        self.lstm_hidden_dim = 200
        self.attention_size = 100
        self.load_basic_info()#加载词汇表等基本信息
        model_config = config_loaders.model_config_loader()
        self.max_text_len = model_config['max_text_len'] + 2
        self.max_question_len = model_config['max_question_len'] + 2
        self.build_graph()#构建计算图

    def load_basic_info(self):
        self.dp = DataProcessor()
        self.dp.load_id_info()
        self.char_id_map = self.dp.char_id_map

    
    def init_inputs_output(self):
        self.text_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_text_len], name='char_ids')#模型的输入，字的id列表，不定长
        self.text_wordseg_tag_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_text_len], name="wordseg_tag_ids")
        self.text_seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None, ], name='seq_lengths')#每个句子的长度
        
        self.question_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_question_len], name='char_ids')#模型的输入，字的id列表，不定长
        self.question_wordseg_tag_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_question_len], name="wordseg_tag_ids")
        self.question_seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None, ], name='seq_lengths')#每个句子的长度
        self.batch_size = tf.shape(self.text_char_ids)[0]
        #输出   
        self.if_no_answer = tf.placeholder(dtype=tf.int32, shape=[None, ], name='seq_lengths')#每个句子的长度
        self.if_no_answer_onehot = tf.nn.embedding_lookup(params=self.if_no_answer_embedding, ids=self.if_no_answer,\
                                                 name="if_no_answer_embeddings")
        self.answer_spam_onthot_sparse = tf.placeholder(dtype=tf.int32, shape=[None, None], name='relation_dist_matrix')#待检查的实体对标签   
        self.answer_span_matrix = self.get_answer_span_onehot(self.answer_spam_onthot_sparse, \
                                                        self.batch_size)
        #超参数
        self.dropout =  tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        
    def build_graph(self):
        self.init_char_tag_encoding()#初始化字向量,ner标签onehot encoding, entity pair的独热编码
        #输入
        self.init_inputs_output()#配置模型的输入和输出
        text_char_embedding_vectors = tf.nn.embedding_lookup(params=self.embeddings, ids=self.text_char_ids,\
                                                 name="char_embeddings")#从词向量矩阵中，为词语找到对应的词向量，形成序列
        text_wordseg_tag_feature = tf.nn.embedding_lookup(params=self.wordseg_tag_onehot, ids=self.text_wordseg_tag_ids)
        text_all_features = tf.concat([text_char_embedding_vectors, text_wordseg_tag_feature], axis=-1)

        question_char_embedding_vectors = tf.nn.embedding_lookup(params=self.embeddings, ids=self.question_char_ids,\
                                                 name="char_embeddings")#从词向量矩阵中，为词语找到对应的词向量，形成序列
        question_wordseg_tag_feature = tf.nn.embedding_lookup(params=self.wordseg_tag_onehot, ids=self.question_wordseg_tag_ids)
        question_all_features = tf.concat([question_char_embedding_vectors, question_wordseg_tag_feature], axis=-1)
        text_encode = self.BiLSTM_layer(text_all_features, self.text_seq_lengths, layer_num="encoder")#编码器。将字特征序列输入到LSTM中
#         text_encode = self.BiLSTM_layer(text_encode, self.text_seq_lengths, layer_num="encoder1")#编码器。将字特征序列输入到LSTM中
        question_encode = self.BiLSTM_layer(question_all_features, self.question_seq_lengths, layer_num="encoder")#解码器
#         question_encode = self.BiLSTM_layer(question_encode, self.question_seq_lengths, layer_num="encoder1")#解码器


        
        question_encode_with_text = self.encode_Q_with_T(text_encode, question_encode)
        text_encode_with_question = self.encode_T_with_Q(text_encode, question_encode)
        text_encode = tf.concat([text_encode, text_encode_with_question], axis=-1)
        question_encode = tf.concat([question_encode, question_encode_with_text], axis=-1)
        text_encode = self.BiLSTM_layer(text_encode, self.text_seq_lengths, layer_num="encoder21")#解码器
        question_encode = self.BiLSTM_layer(question_encode, self.question_seq_lengths, layer_num="encoder22")#解码器


        #使用指针网络计算答案输出
        answer_span_dist_logits_matrix = self.ptr_network(text_encode, question_encode, self.batch_size)#使用sigmoid判断实体关系类型
        self.pred_answer_span_dist_matrix = tf.nn.softmax(answer_span_dist_logits_matrix)
#         self.pred_answer_span_onehot = tf.round(self.pred_answer_span_dist_matrix, name="pred")
        self.pred_answer_span = tf.argmax(self.pred_answer_span_dist_matrix, 2)
        penalty = self.get_penalty(tf.slice(self.pred_answer_span, [0, 0], [-1, 0]), 
                                   tf.slice(self.pred_answer_span, [0, 1], [-1, 1]))#用relu激活函数，对“左大右小"的反常情况进行惩罚
        
        #计算文档是否蕴含答案
        span_info = tf.layers.flatten(answer_span_dist_logits_matrix)
#         span_info = tf.layers.dense(span_info, units=50, activation=tf.nn.sigmoid)
        pred_if_no_answer_onehot_logits = tf.layers.dense(span_info, units=2)
        pred_if_no_answer_onehot_probs = tf.nn.softmax(pred_if_no_answer_onehot_logits)
        self.pred_if_no_answer_onehot_labels = tf.argmax(pred_if_no_answer_onehot_probs)
        print('penalty', penalty)
        self.span_loss = tf.reduce_mean(\
                 tf.nn.softmax_cross_entropy_with_logits(\
                logits=answer_span_dist_logits_matrix,labels=self.answer_span_matrix), name="span_loss") +\
                                                                                                   penalty/100
        self.if_no_answer_loss = tf.reduce_mean(\
                 tf.nn.softmax_cross_entropy_with_logits(\
                logits=pred_if_no_answer_onehot_logits,labels=self.if_no_answer_onehot), name="if_no_answer_loss")
        self.loss = self.span_loss + self.if_no_answer_loss
        print('self.loss', self.loss)

        with tf.variable_scope("optimizer"): 
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#             self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        mirrored_strategy = tf.distribute.MirroredStrategy()
        
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def get_penalty(self, left_index, right_index):
        loss = tf.reduce_sum(tf.nn.tanh(tf.nn.relu(tf.cast(left_index - right_index, dtype=tf.float32))))
        loss += tf.reduce_sum(tf.nn.tanh(tf.nn.relu(tf.cast( - left_index + right_index - 50, dtype=tf.float32))))#如果右边界超出左边界太多，也要惩罚
        return loss

   
    def encode_Q_with_T(self, text_encode, question_encode):
        q_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        q_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))
        k_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        k_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))

        text_encode = tf.tensordot(text_encode, q_w, axes=1) + q_b
        question_encode = tf.tensordot(question_encode, k_w, axes=1) + k_b

        attention_weight = tf.matmul(question_encode, text_encode, transpose_b=True)
        attention_weight = tf.nn.sigmoid(attention_weight)

        w = tf.Variable(tf.random_normal([self.attention_size, self.max_question_len], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.max_question_len], stddev=0.01))        
        question_encode_v = tf.tensordot(question_encode, w, axes=1) + b
        attention_weight = tf.matmul(question_encode_v, attention_weight)
        
        attention_weight = tf.nn.dropout(attention_weight, self.dropout)
        return attention_weight
    
    def encode_T_with_Q(self, text_encode, question_encode):
        q_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        q_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))
        k_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        k_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))
        
#         w = tf.Variable(tf.random_normal([self.attention_size, self.attention_size], stddev=0.01))
#         b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))
        
        text_encode = tf.tensordot(text_encode, q_w, axes=1) + q_b
        question_encode = tf.tensordot(question_encode, k_w, axes=1) + k_b
#         question_encode_v = tf.tensordot(question_encode, w, axes=1) + b
#         question_encode = tf.nn.sigmoid(question_encode)
        attention_weight = tf.matmul(text_encode, question_encode, transpose_b=True)
        attention_weight = tf.nn.sigmoid(attention_weight)
        
        w = tf.Variable(tf.random_normal([self.attention_size, self.max_question_len], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.max_question_len], stddev=0.01))        
        text_encode_v = tf.tensordot(text_encode, w, axes=1) + b
        attention_weight = tf.matmul(text_encode_v, attention_weight, transpose_b=True)
        
        attention_weight = tf.nn.dropout(attention_weight, self.dropout)
        return attention_weight
        
    #使用指针网络，基于biLSTM得到的语义向量序列，得到答案在原文中的span
    def ptr_network(self, text_encode, question_encode, batch_size):
        q_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        q_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))
        k_w = tf.Variable(tf.random_normal([self.lstm_hidden_dim*2, self.attention_size], stddev=0.01))
        k_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.01))

        text_encode = tf.tensordot(text_encode, q_w, axes=1) + q_b
        question_encode = tf.tensordot(question_encode, k_w, axes=1) + k_b

        attention_weight = tf.matmul(question_encode, text_encode, transpose_b=True)
        
        w = tf.Variable(tf.random_normal([self.attention_size, self.max_question_len], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.max_question_len], stddev=0.01))        
        text_encode_v = tf.tensordot(text_encode, w, axes=1) + b
        
        #使用rnn对问题和文档进行总结
        print('attention_weight', attention_weight)
#         answer_span_dist = self.BiLSTM_layer_for_answer(attention_weight, batch_size, self.max_text_len)
        answer_span_dist = tf.slice(attention_weight, [0, 0, 0], [-1, 2, self.max_text_len])
        return answer_span_dist

    def BiLSTM_layer_for_answer(self, lstm_inputs, batch, unit_num=None, lstm_layers=1):
        if unit_num==None: unit_num=self.lstm_hidden_dim
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(unit_num) for i in range(lstm_layers)])
        init_state = cell.zero_state(batch, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, lstm_inputs, \
                        sequence_length=self.question_seq_lengths, initial_state=init_state, dtype=tf.float32)
        print('output_rnn', output_rnn)
        output = tf.slice(output_rnn, [0, self.max_question_len-2, 0], [batch, 2, self.max_text_len])
        return output
    
    def get_answer_span_onehot(self, answer_spam_onthot_sparse, batch_size):
        answer_span_distribution_sparse_indexes = tf.reshape(answer_spam_onthot_sparse, shape=[-1, 2])
        answer_span_dist_matrix = tf.sparse_to_dense(answer_span_distribution_sparse_indexes, \
                                                  [batch_size, self.max_text_len*2], 1.0, 0)
        answer_span_matrix = tf.reshape(answer_span_dist_matrix, shape=[batch_size, 2, self.max_text_len])
        return answer_span_matrix

    def init_char_tag_encoding(self, use_pretrained_embeddings=True, embedding_dim=100):
        print("初始化字向量。")
        if self.mode=='train':#如果是训练，加载预训练好的，或者随机初始化。
            if use_pretrained_embeddings==True:
                print("读取预训练的词向量")
                embeddings = pickle.load(open(run_time.PATH_PRETRAINED_EMBEDDINGS, 'rb'))
            else:
                print("随机初始化一份词向量")
                embeddings = np.float32(np.random.uniform(-0.5, 0.5, \
                                                               (len(self.char_id_map), embedding_dim)))
        else:#如果是其他模式，加载模型自己训练得到的词向量即可
            print("加载模型自己的词向量")
            embeddings = pickle(open(run_time.PATH_EMBEDDINGS, 'rb')) 
        #将初始化后的嵌入向量添加到计算图找那个
        with tf.variable_scope("words"):
            self.embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=True,\
                                           name="char_embeddings")#词向量是一个变量；当然也可以使用trainable冻结
            
        #分词标签的独热编码
        wordseg_tag_onehot_np = np.eye(7, dtype=np.float32)
        self.wordseg_tag_onehot = tf.Variable(wordseg_tag_onehot_np, dtype=tf.float32, trainable=False, name="wordseg_tag")

        if_no_answer_onehot_np = np.eye(2, dtype=np.float32)
        self.if_no_answer_embedding = tf.Variable(if_no_answer_onehot_np, dtype=tf.float32, trainable=False, name="wordseg_tag")
            
                        
    def BiLSTM_layer(self, lstm_inputs, seq_lengths, layer_num=0, unit_num=None):
        if unit_num==None: unit_num=self.lstm_hidden_dim
        with tf.variable_scope("bilstm_" + str(layer_num), reuse=tf.AUTO_REUSE):
            cell_fw, cell_bw = LSTMCell(unit_num), LSTMCell(unit_num)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=lstm_inputs,\
                 sequence_length=seq_lengths, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        return output


    def get_feed_dict(self, data_batch, if_trian=True, learning_rate=0.001):
        if if_trian: dropout = 0.5
        else: dropout = 1.0
#         print(data_batch[1])
#         print("文本数据形状", np.array(data_batch[4]).shape)
#         print("序列长度数据形状", np.array(data_batch[4]).shape, max(data_batch[6]))
        feed_dict = {self.text_char_ids: data_batch[1],
                     self.text_wordseg_tag_ids: data_batch[2],
                     self.text_seq_lengths: data_batch[3],
                     self.question_char_ids: data_batch[4],
                     self.question_wordseg_tag_ids: data_batch[5],
                     self.question_seq_lengths: data_batch[6],
                     self.if_no_answer: data_batch[7],
                    self.answer_spam_onthot_sparse: data_batch[-1],
                    self.dropout: dropout,
                    self.batch_size: len(data_batch[0]),
                    self.learning_rate: learning_rate
                     }
        return feed_dict
    
    #训练
    def fit(self, epoch_num=10000):
        loss_batch = []
        init_learning_rate, decay_rate = 0.0003, 1
        learning_rate_plan = [0.005, 0.0005]

        sample_list = self.dp.get_train_data(run_time.PATH_CORPUS_SAMPLE, if_train=True)
        import copy
#         test_sample_list, sample_list = sample_list[:100], copy.deepcopy(sample_list[:100])
        test_sample_list = self.dp.get_train_data(run_time.PATH_CORPUS_SAMPLE)
        test_data_batch_list = self.dp.preprocess_batches(test_sample_list)
        print("开始训练, 训练数据量是", len(sample_list))
        for epoch in range(1, epoch_num):
            step = 0
            random.shuffle(sample_list)
            data_batch_list = self.dp.preprocess_batches(sample_list)
#             print("训练集大小", len(data_batch_list))
            for a_batch_of_sample in data_batch_list:
                step += 1
                _, loss, pred_answer_span = self.sess.run((self.train, self.loss, self.pred_answer_span), \
                            feed_dict=self.get_feed_dict(a_batch_of_sample, learning_rate=init_learning_rate*decay_rate**epoch))
                loss_batch.append(loss)
                if step%100==0:
                    print('epoch', epoch, 'step', step, "/", len(data_batch_list), "。 loss is", np.mean(loss_batch), int(time.time()))
                    loss_batch = []
#                 if step%1==0:
#                     for i in range(min(10, len(pred_answer_span))):
#                         print('self.pred_answer_span', pred_answer_span[i], \
#                               a_batch_of_sample[0][i]['context'][pred_answer_span[i][0]-1: pred_answer_span[i][1]-1])
#                         print("real span", a_batch_of_sample[0][i]['answers'])
# #                     
            if epoch%10==0:
                self.evaluate(epoch, test_data_batch_list)
    
    def predict(self, batches):
        spans_list = []
        loss_list = []
        for a_batch_sample in batches:
            loss, pred_answer_span = self.sess.run((self.loss, self.pred_answer_span), \
                                        feed_dict=self.get_feed_dict(a_batch_sample, if_trian=False))
            
            spans_list.append(pred_answer_span)
            loss_list.append(loss)
        return spans_list, np.mean(loss_list)
            
    def evaluate(self, epoch, test_data_list):
        spans_list, loss = self.predict(test_data_list)
        
        right_num = 0
        total_entity_num = 0
        found_entity_num = 0
        for i in range(len(test_data_list)):
            real_batch = test_data_list[i][0]
#             print(real_batch)
            for j in range(len(real_batch)):
                if real_batch[j]['answers']!=None and len(real_batch[j]['answers'])>0:
                    real_answer = real_batch[j]['answers'][0]['text']
                    total_entity_num += 1
                else:
                    real_answer = ""
                
                span = spans_list[i][j]
                pred_answer = real_batch[j]['context'][span[0]-1: span[1]-1]
#                 print('real_answer', real_answer, 'pred_answer', pred_answer, span)
                if len(pred_answer)>0: found_entity_num += 1
                if len(real_answer)>0 and real_answer==pred_answer: right_num += 1   
                
        recall = right_num/total_entity_num if total_entity_num>0 else 0
        precision = right_num/found_entity_num if found_entity_num>0 else 0
        f1_score = 2*recall*precision/(precision + recall) if precision + recall>0 else 0
        print('epoch', epoch, "的测试结果是:")
        print("测试集中的loss为", loss, "召回率是", recall, "精度是", precision, "f1-score是", f1_score)
        with open("test_result.txt", 'a', encoding='utf8') as f:
            f.write("第" + str(epoch) + "轮的效果是:\n")
            f.write("召回率是" + str(recall)[:6] + " 精度是" + str(precision)[:6] + \
                     " F1-score是" + str(f1_score)[:6]+ "\n")

if __name__ == '__main__':
#     print("数据批数是", len(train_data))
    model = Model(mode="train")
    model.fit()

