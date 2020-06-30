'''
Created on 2020��4��6��

@author: Administrator
'''
#使用bert as  service做特征提取器，然后使用神经网络分类
# 0.8638787
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import pickle
from config import run_time
from utils.corpus_loader import CorpusLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from bert_serving.client import BertClient
import json
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

class FeatureEngineering():
    
    def __init__(self, sample=None):
        self.sample = sample
        self.vocab_size = 4000
        self.token_id_map = {}
        self.id_token_map = {}
        self.bert_client = BertClient(ip='192.168.1.104')
        
    
    def process(self):
        texts, labels_all, onehot_labels = CorpusLoader().load_toutiao_cat_data(sample=self.sample)
        sentence_vectors = []
        labels = []
        text_batch, label_batch = [], []
        batch_size = 200
        for i in range(len(texts)):
            text = texts[i]
            label = labels_all[i]
            text_batch.append(text)
            label_batch.append(label)
#             if i==300: break
            if len(text_batch)==batch_size:
                print(i, len(texts))
                vector_batch =  self.bert_client .encode(text_batch)
                sentence_vectors += list(vector_batch)
                labels += label_batch
                text_batch, label_batch = [], []
        self.save_corpus(sentence_vectors, labels, run_time.PATH_TOUTIAO_TRAINING_DATA_FOR_BERT_V1)
        return texts, labels, onehot_labels

    def save_corpus(self, inputs, labels, target_file):
        print("保存语料")
        pickle.dump({'labels': labels, "inputs": inputs}, open(target_file, 'wb'))
        
    def load_corpus(self):
        print("加载语料")
        data = pickle.load(open(run_time.PATH_TOUTIAO_TRAINING_DATA_FOR_BERT_V1, 'rb'))
        return data
      
class RCNN():
    
    def __init__(self, mode="train"):
        self.mode = mode
        self.load_model_configs()
        self.init_graph()
        if self.mode=="work":
            self.load_model()#加载模型时，会把保存的参数填到计算图中
    
    def load_model_configs(self):
        model_infos = json.load(open(run_time.PATH_MODEL_RNN_CNN_INFOS, 'r', encoding='utf8'))
        self.class_num = model_infos['class_num']
        self.batch_size = 500#model_infos['batch_size']
        self.max_epoch = model_infos['max_epoch']
        self.text_vector_dim = 768
        
    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(run_time.PATH_MODEL_RNN_CNN_CHECK_POINT_DIR) 
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
       
    def init_graph(self):
        #定义输入
        self.text_vectors = tf.placeholder(tf.float32, shape=[None, self.text_vector_dim], name="input_is_token_ids")
        self.text_class_list = tf.placeholder(tf.float32, shape=[None, self.class_num], name="text_label")
        self.dropout = tf.placeholder(tf.float32)
        
        #使用softmax层得到类别标签
        res1 = tf.layers.dense(self.text_vectors, 100, activation=tf.nn.tanh)
        res1_ = tf.nn.dropout(res1, keep_prob=self.dropout)
        res2 = tf.layers.dense(res1_, 50, activation=tf.nn.tanh)
        final_res = tf.layers.dense(res2, self.class_num, activation=tf.nn.tanh)
        self.prob_vectors = tf.nn.softmax(final_res)
        #计算损失值和准确率
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                    labels=self.text_class_list, logits=self.prob_vectors))
        
        real_labels = tf.argmax(self.text_class_list, axis=-1, output_type=tf.int32)
        pred_labels = tf.argmax(self.prob_vectors, axis=-1, output_type=tf.int32)
        print("real_labels", real_labels, 'pred_labels', pred_labels)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_labels, real_labels)))
        
        self.train = tf.train.AdamOptimizer(0.0002).minimize(self.loss)
        
        self.sess = tf.Session()
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
    
    def data2batches(self, X, Y):
        indexes = list(range(Y.shape[0]))
        random.shuffle(indexes)
        X_batches, Y_batches = [], []
        X_batch, Y_batch = [], []
        for index in indexes:
            X_batch.append(X[index])
            Y_batch.append(Y[index])
#             print("积累", len(Z_batch), self.batch_size)
            if len(X_batch)==self.batch_size:
                X_batches.append(X_batch)
                Y_batches.append(Y_batch)
                X_batch, Y_batch, Z_batch = [], [], []
        if len(Y_batch)!=0:
            X_batches.append(X_batch)
            Y_batches.append(Y_batch)
        return X_batches, Y_batches
    
    def get_input_dict(self, padden_token_ids_list, onehot_labels, dropout=1):
        input_dict = {self.text_vectors: np.array(padden_token_ids_list), 
                      self.text_class_list: np.array(onehot_labels), \
                    self.dropout: dropout }
#         print(np.array(padden_token_ids_list).shape)
#         print(np.array(onehot_labels).shape)
#         print(np.array(token_seq_len_list).shape)
        return input_dict
    
    def fit(self, inputs, onehot_labels):
        
        training_inputs, test_inputs, onehot_labels, test_labels = train_test_split(inputs, onehot_labels, test_size=0.05)
        test_input_dict = self.get_input_dict(test_inputs, test_labels, dropout=1)
        for epoch in range(self.max_epoch):
            train_X_batches, train_Y_batches = self.data2batches(training_inputs, onehot_labels)
            for batch_no in range(len(train_Y_batches)):
                X_batch, y_batch = train_X_batches[batch_no], train_Y_batches[batch_no]
#                 print(len(X_batch), len(y_batch), len(seq_lens_batch), y_batch)
                input_dict = self.get_input_dict(X_batch, y_batch, dropout=0.5)
#                 print(input_dict)
                [_, loss, accuracy] = self.sess.run([self.train, self.loss, \
                                                                      self.accuracy], feed_dict=input_dict)
                if batch_no%5000==0:
                    [loss, accuracy] = self.sess.run([self.loss, self.accuracy], feed_dict=test_input_dict)
                    print("epoch",epoch, "batch_no", batch_no, "loss", loss, 'accuracy', accuracy)
                
    def predict(self, text_list):
        pass

from sklearn.preprocessing import OneHotEncoder
def tradicional_methods():
    FE = FeatureEngineering()
#     FE.process()
    data = FE.load_corpus()
    inputs, labels = data['inputs'], data['labels']
    print("inputs 的形状是", len(inputs[0]))
    OE = OneHotEncoder()
    labels = np.array(labels).reshape([-1, 1])
    
    OE.fit(labels)
    labels = OE.transform(labels)
    labels = labels.toarray()
    #模型
    model = RCNN()
    model.fit(inputs, labels)
if __name__ == '__main__':
    tradicional_methods()
    
    
    