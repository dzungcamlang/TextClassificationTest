'''
Created on 2020��4��6��

@author: Administrator
'''
#使用RNN和CNN来做文本分类
#0.83
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from utils.corpus_loader import CorpusLoader
from config import run_time
import pickle
import numpy as np
import tensorflow as tf
import json
import random
from tensorflow.contrib.layers.python.layers import initializers
from sklearn.model_selection import train_test_split
from utils.splitSentence import getSentences
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class FeatureEngineering():
    
    def __init__(self, sample=None):
        self.sample = sample
        self.vocab_size = 4000
        self.token_id_map = {}
        self.id_token_map = {}
        
    
    def process(self):
        texts, _, onehot_labels = CorpusLoader().load_toutiao_cat_data(sample=self.sample)
        self.get_feature_names(texts)
        
        pretrained_embedding_vectors = self.extract_pretrained_embedding()
#         print('pretrained_embedding_vectors', pretrained_embedding_vectors)
        pickle.dump({"token_id_map": self.token_id_map, 
             "id_token_map": self.id_token_map,
             "vocab_size": self.vocab_size,
             "pretrained_embedding_vectors": pretrained_embedding_vectors}, 
             open(run_time.PATH_TOUTIAO_DATA_TOKENS, 'wb'))
        return texts, _, onehot_labels
    

    
    def get_feature_names(self, texts):
        term_freq_map = {}
        for text in texts:
            for hanzi in text:
                term_freq_map[hanzi] = term_freq_map.get(hanzi, 0) + 1

        features = sorted(term_freq_map.items(), key=lambda x: x[1])
        features = list(map(lambda x: x[0], features))
        features = features[-self.vocab_size :]
        
        self.token_id_map = {"<PADDING>": 0, "<UNKNOWN>": 1}
        for i in range(len(features)):
            self.token_id_map[features[i]] = len(self.token_id_map)
            
        for token, id in self.token_id_map.items():
            self.id_token_map[id] = token
        self.vocab_size = len(self.token_id_map)
    
    def extract_pretrained_embedding(self):
        lines = list(open(run_time.PATH_HANZI_EMBEDDING_VECTORS, 'r', encoding='utf8').readlines())
        vectors = [None for _ in range(self.vocab_size)]
        for line in lines[1:]:
            slices = line.split(" ")
            hanzi = slices[0]
            if hanzi in self.token_id_map:
                vector = list(map(lambda x: float(x), slices[1:]))
                vectors[self.token_id_map[hanzi]] = vector
        
        vectors = list(map(lambda x: x if x!=None else [0]*len(vector), vectors))
        return vectors
        

class RCNN():
    
    def __init__(self, mode="train"):
        self.mode = mode
        self.load_token_infos()
        self.load_model_configs()
        self.init_graph()
        if self.mode=="work":
            self.load_model()#加载模型时，会把保存的参数填到计算图中
    
    def load_token_infos(self):
        token_infos = pickle.load(open(run_time.PATH_TOUTIAO_DATA_TOKENS, 'rb'))
        self.vocab_size, self.id_token_map, self.token_id_map, self.pretrained_embeddings = \
                            token_infos["vocab_size"], token_infos["id_token_map"], \
                            token_infos["token_id_map"], token_infos['pretrained_embedding_vectors']
    
    def load_model_configs(self):
        model_infos = json.load(open(run_time.PATH_MODEL_RNN_CNN_INFOS, 'r', encoding='utf8'))
        self.embedding_dim = model_infos['embedding_dim']
        self.embedding_trainable = True if model_infos['embedding_trainable']==1 else False
        self.use_pretrained_embedding = True if model_infos['use_pretrained_embedding']==1 else False
        self.max_seq_len = model_infos['max_seq_len']
        self.class_num = model_infos['class_num']
        self.NN_type = model_infos['NN_type']
        self.LSTM_unit_num = model_infos['LSTM_unit_num']
        self.batch_size = model_infos['batch_size']
        self.max_epoch = model_infos['max_epoch']
        self.attention_size = model_infos['attention_size']
        
    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(run_time.PATH_MODEL_RNN_CNN_CHECK_POINT_DIR) 
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def BiLSTM_layer_high_way(self, inputs, layer_name="name"):
        with tf.variable_scope(layer_name):
            directions = ["forward", "backward"]
            lstm_map = {}
            for direction in directions:
                lstm_map[direction] = tf.nn.rnn_cell.LSTMCell(num_units=50, forget_bias=0.1, \
                                                              state_is_tuple=True, \
                                                              use_peepholes=True, \
                                                              initializer=initializers.xavier_initializer())
                
    
    #             lstm_map[direction] = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=1-self.dropout)
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_map['forward'], lstm_map['backward'],\
                                                                    inputs, \
                                                                    sequence_length=self.token_seq_len_list,\
                                                                    dtype=tf.float32)
        return tf.concat(outputs, axis=2)
        
    def BiLSTM_layer(self, inputs, layer_name="name"):
        with tf.variable_scope(layer_name):
            directions = ["forward", "backward"]
            lstm_map = {}
            for direction in directions:
                lstm_map[direction] = tf.nn.rnn_cell.LSTMCell(num_units=self.LSTM_unit_num, forget_bias=0.1, \
                                                              state_is_tuple=True, \
                                                              use_peepholes=True, \
                                                              initializer=initializers.xavier_initializer())
                
    
    #             lstm_map[direction] = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=1-self.dropout)
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_map['forward'], lstm_map['backward'],\
                                                                    inputs, \
                                                                    sequence_length=self.token_seq_len_list,\
                                                                    dtype=tf.float32)
        return tf.concat(outputs, axis=2)
    
    #针对序列数据的自注意力机制
    def self_attention(self, inputs):
        input_seq = tf.concat(inputs, 2)
        print("input_seq", input_seq)
        q_w = tf.Variable(tf.random_normal([self.LSTM_unit_num*2, self.attention_size], stddev=0.1))
        q_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        k_w = tf.Variable(tf.random_normal([self.LSTM_unit_num*2, self.attention_size], stddev=0.1))
        k_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
#         v_w = tf.Variable(tf.random_normal([self.LSTM_unit_num*2, self.attention_size], stddev=0.1))
#         v_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        query = tf.tensordot(input_seq, q_w, axes=1) + q_b
        key = tf.tensordot(input_seq, k_w, axes=1) + k_b
#         value = tf.tensordot(input_seq, v_w, axes=1) + v_b
        
        query_key = tf.matmul(query, key, transpose_b=True)
        query_key_ = tf.nn.softmax(query_key)
        qkv = tf.matmul(query_key_, inputs)
        return qkv
        
        
    #https://blog.csdn.net/huanghaocs/article/details/85255227
    def attention(self, inputs, attention_size=100, time_major=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:  # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value 
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        #the result has (B, T, D) shape
        output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
        output = tf.add(output,inputs)#改动点
#         output = tf.transpose(output, perm=[1, 0, 2])
        return output
        
    def CNN_layer(self):
        layer_1 = tf.layers.conv2d(tf.reshape(self.token_embeddings, [-1, self.max_seq_len, self.embedding_dim, 1]), filters=20, kernel_size=[5, self.embedding_dim], \
                                                     strides=4, padding="SAME", activation=tf.nn.tanh)
        layer_1_pool = tf.layers.max_pooling2d(layer_1, pool_size=[5, 5], strides=5)
        
        layer_2 = tf.layers.conv2d(layer_1_pool, filters=10, kernel_size=[3, 1], \
                                                     strides=2, padding="SAME", activation=tf.nn.tanh)
        layer_2_pool = tf.layers.max_pooling2d(layer_2, pool_size=[3, 3], strides=2)
        return layer_2_pool
        
    def init_graph(self):
        #初始化字嵌入向量
        if self.use_pretrained_embedding==True:
            self.embeddings = tf.Variable(self.pretrained_embeddings, trainable=self.embedding_trainable)
        else:
            self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_dim], 0, 0.1, tf.float32), trainable=True)

        #定义输入
        self.padden_token_ids_list = tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name="input_is_token_ids")
        print('self.padden_token_ids_list', self.padden_token_ids_list)
        self.token_seq_len_list = tf.placeholder(tf.int32, shape=[None], name="length_of_each_text")
        self.text_class_list = tf.placeholder(tf.float32, shape=[None, self.class_num], name="text_label")
        self.dropout = tf.placeholder(tf.float32)
        
        #将token序列转换为字向量序列
        token_embeddings = tf.nn.embedding_lookup(self.embeddings, self.padden_token_ids_list)
        self.token_embeddings = tf.nn.dropout(token_embeddings, keep_prob=self.dropout)
        #用RNN或者CNN对字向量序列进行处理，提取特征
        if self.NN_type=="BiLSTM":
            print("LSTM working")
            features = self.BiLSTM_layer(self.token_embeddings, 'layer1')
            features = self.BiLSTM_layer(features, 'layer2')
#             features = self.attention(features)
#             features = self.self_attention(features)
        else:# self.NN_type=="CNN":
            features = self.CNN_layer()
        print("features", features)
        #使用softmax层得到类别标签
        features_flatten = tf.layers.flatten(features)
#         res1 = tf.layers.dense(features_flatten, 100, activation=tf.nn.tanh)
        res2 = tf.layers.dense(features_flatten, self.class_num, activation=tf.nn.tanh)
        self.prob_vectors = tf.nn.softmax(res2)
        #计算损失值和准确率
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                    labels=self.text_class_list, logits=self.prob_vectors))
        
        real_labels = tf.argmax(self.text_class_list, axis=-1, output_type=tf.int32)
        pred_labels = tf.argmax(self.prob_vectors, axis=-1, output_type=tf.int32)
        print("real_labels", real_labels, 'pred_labels', pred_labels)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_labels, real_labels)))
        
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        gpu_options.allow_growth = True
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.initialize_all_variables())
        #设置模型保存器
        self.saver = tf.train.Saver()
    


    def text2padden_token_ids(self, texts):
        token_ids_list = []
        seq_list = []
        for text in texts:
            token_ids = []
            for hanzi in text:
                if hanzi in self.token_id_map: token_ids.append(self.token_id_map[hanzi])
                else: token_ids.append(1)
            if len(token_ids)>self.max_seq_len: token_ids = token_ids[:self.max_seq_len]
            else: token_ids = token_ids + [0]*(self.max_seq_len - len(token_ids))
            token_ids_list.append(token_ids)#补全后的token id序列
            seq_list.append(len(token_ids))#token序列长度
        return token_ids_list, seq_list
    
    def data2batches(self, X, Y, Z):
        indexes = list(range(len(Y)))
        random.shuffle(indexes)
        X_batches, Y_batches, Z_batches = [], [], []
        X_batch, Y_batch, Z_batch = [], [], []
        for index in indexes:
            X_batch.append(X[index])
            Y_batch.append(Y[index])
            Z_batch.append(Z[index])
#             print("积累", len(Z_batch), self.batch_size)
            if len(X_batch)==self.batch_size:
                X_batches.append(X_batch)
                Y_batches.append(Y_batch)
                Z_batches.append(Z_batch)
                X_batch, Y_batch, Z_batch = [], [], []
        if len(Y_batch)!=0:
            X_batches.append(X_batch)
            Y_batches.append(Y_batch)
            Z_batches.append(Z_batch)            
        return X_batches, Y_batches, Z_batches
    
    def get_input_dict(self, padden_token_ids_list, onehot_labels, token_seq_len_list, dropout=1):
        input_dict = {self.padden_token_ids_list: np.array(padden_token_ids_list), 
                      self.text_class_list: np.array(onehot_labels), \
                      self.token_seq_len_list: np.array(token_seq_len_list),
                    self.dropout: dropout }
        return input_dict
    
    #数据增强
    def data_aug(self, texts, labels):
        new_texts, new_labels = [], []
        for i in range(len(texts)):
            sentences = getSentences(texts[i])
            random.shuffle(sentences)
            new_text = "".join(sentences)
            
            new_texts.append(texts[i])
            new_labels.append(labels[i])
            new_texts.append(new_text)
            new_labels.append(labels[i])
        return new_texts, new_labels
    
    def fit(self, text_list, onehot_labels):
        
        text_list, test_texts, onehot_labels, test_onehot_labels = train_test_split(text_list, onehot_labels, test_size=0.05)
#         text_list, onehot_labels = self.data_aug(text_list, onehot_labels)
        print("数据增强后训练集大小是", len(text_list))
        token_ids_list, seq_lens = self.text2padden_token_ids(text_list)
        
        #测试集处理
        test_token_ids_list, test_seq_lens = self.text2padden_token_ids(test_texts)
        test_X_batches, test_Y_batches, test_seq_len_batches = self.data2batches(test_token_ids_list, test_onehot_labels, test_seq_lens)
        test_input_dicts = [self.get_input_dict(test_X_batches[i], test_Y_batches[i], test_seq_len_batches[i], dropout=1) for i in range(len(test_X_batches))]
        for epoch in range(self.max_epoch):
            train_X_batches, train_Y_batches, seq_len_batches = self.data2batches(token_ids_list, onehot_labels, seq_lens)
            train_loss_list = []
            for batch_no in range(len(seq_len_batches)):
                X_batch, y_batch, seq_lens_batch = train_X_batches[batch_no], train_Y_batches[batch_no], seq_len_batches[batch_no]
#                 print(len(X_batch), len(y_batch), len(seq_lens_batch), y_batch)
                input_dict = self.get_input_dict(X_batch, y_batch, seq_lens_batch, dropout=0.5)
                [_, loss, accuracy] = self.sess.run([self.train, self.loss, \
                                                                      self.accuracy], feed_dict=input_dict)
                train_loss_list.append(loss)
                if batch_no%200==0: 
                    print("epoch",epoch, "batch_no", batch_no, "loss", np.mean(train_loss_list))
                    train_loss_list = []
#                 if batch_no%2000==0:
            losses, accs = [], []
            for batch in test_input_dicts:
                [loss, accuracy] = self.sess.run([self.loss, self.accuracy], feed_dict=batch)
                losses.append(loss)
                accs.append(accuracy)
            print("epoch",epoch, "batch_no", batch_no, "loss", np.mean(losses), 'accuracy', np.mean(accs))
                
    def predict(self, text_list):
        pass

def RNN_CNN():
    FE = FeatureEngineering()
    texts, _, onehot_labels = FE.process()
    
    #模型
    model = RCNN()
    model.fit(texts, onehot_labels)
if __name__ == '__main__':
    RNN_CNN()
    