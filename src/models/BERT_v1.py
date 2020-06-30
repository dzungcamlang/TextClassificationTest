'''
Created on 2020��4��6��

@author: Administrator
'''
#使用bert as  service做特征提取器，然后使用传统算法分类
# 0.8486722425509671
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
            
class TratitionalModel():
    
    def __init__(self):
        self.model = MLPClassifier([50, 10])#RFC(n_estimators=40,n_jobs=7)#BernoulliNB()#
    
    def fit(self, inputs, labels):
        self.model.fit(inputs, labels)
        
    def evaluate(self, inputs, labels):
        predictions = self.model.predict(inputs)
        accuracy = metrics.accuracy_score(labels, predictions)
        print("准确率是", accuracy)
    

def tradicional_methods():
    FE = FeatureEngineering()
#     FE.process()
    
    data = FE.load_corpus()
    model = TratitionalModel()
    print(data.keys())
    inputs, labels = data['inputs'], data['labels']
#     for line in inputs: print(sum(line))
    inputs = np.array(inputs)
    labels = list(map(lambda x: [x], labels))
    labels = np.array(labels)
    inputs, test_inputs, labels, test_labels = train_test_split(inputs, labels, test_size=0.50)
    print("input的形状是", inputs.shape, len(labels))
    print("训练")
    model.fit(inputs, labels)
    print("测试")
    model.evaluate(test_inputs, test_labels)
if __name__ == '__main__':
    tradicional_methods()
    
    
    