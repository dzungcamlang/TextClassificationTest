'''
Created on 2020��4��6��

@author: Administrator
'''
#用传统算法做文本分类
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
import jieba

class FeatureEngineering():
    
    def __init__(self, sample=None):
        self.feature_ids = {}
        self.max_dim = 2000
        self.feature_num = -1
        self.sample = sample
        self.idf_map = {}
        self.stop_word_set = set({})
    
    def get_ngrams(self, texts, N=2):
        ngrams_list = []
        for text in texts:
            ngrams = []
            for i in range(len(text)-N):
                ngrams.append(text[i : i + N])
            ngrams_list.append(ngrams)
        return ngrams_list
    
    def get_words(self, texts):
        words_list = []
        for text in texts:
            words = list(jieba.cut(text))
            words_list.append(words)
        return words_list

    def process(self):
        texts, labels, _ = CorpusLoader().load_toutiao_cat_data(sample=self.sample)
        print("类别个数是", len(set(labels)))
        print("获取ngrams")
        ngrams_list = self.get_words(texts)#self.get_ngrams(texts)
        print("特征选择")
        self.get_feature_names(ngrams_list)
        print("特征工程最后阶段")
        inputs = self.get_input(ngrams_list)
        self.save_corpus(inputs, labels, run_time.PATH_TOUTIAO_DATA_NGRAMS)
    
    def get_input(self, ngrams_list):
        inputs = []
        for ngrams in ngrams_list:
            input = [0 for i in range(self.feature_num)]
            for ngram in ngrams:
                if ngram in self.feature_ids:
                    input[self.feature_ids[ngram]] += 1#*self.idf_map.get(ngram, 0.01)
            inputs.append(input)
        return inputs
        
    def get_feature_names(self, ngrams_list):
#         self.load_stop_words()
        self.get_idf(ngrams_list)#计算idf
        ngram_freq = {}
        for ngrams in ngrams_list:#计算tf
            for ngram in ngrams:
#                 print(ngram, self.stop_word_set)
                if ngram in self.stop_word_set:
#                     print(ngram)
                    continue
                ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        TF_IDF = {}
        for ngram in ngram_freq:#计算tf-idf
            TF_IDF[ngram] = self.idf_map[ngram] * ngram_freq[ngram]
        
        #开始筛选
        features = sorted(TF_IDF.items(), key=lambda x: x[1])
        features = list(map(lambda x: x[0], features))
        if len(features) > self.max_dim:
            upper_index = int(0.001*len(features))
            features = features[-upper_index - self.max_dim: -upper_index]
        for i in range(len(features)):
            self.feature_ids[features[i]] = i
        
        self.feature_num = len(features)
#         print("self.feature_num", self.feature_num)

    def get_idf(self, words_list):
        doc_num = len(words_list)
        doc_freq_map = {}
        for words in words_list:
            for word in set(words):
                doc_freq_map[word] = doc_freq_map.get(word, 0) + 1
        self.idf_map = {}
        for word in doc_freq_map:
            self.idf_map[word] = np.log(doc_num/(doc_freq_map[word] + 1)**0.6)
    
    def load_stop_words(self):
        words = list(open(run_time.PATH_HIT_STOP_WORDS, 'r', encoding='utf8').readlines())
        words = list(map(lambda x: x.replace("\n", ""), words))
        self.stop_word_set = set(words)
    
    def save_corpus(self, inputs, labels, target_file):
        print("保存语料")
        pickle.dump({'labels': labels, "inputs": inputs}, open(target_file, 'wb'))
        
    def load_corpus(self, target_file):
        print("加载语料")
        return pickle.load(open(target_file, 'rb'))
        
class TratitionalModel():
    
    def __init__(self):
        self.model = RFC(n_estimators=10,n_jobs=7)#MLPClassifier([50, 10])#BernoulliNB()#
    
    def fit(self, inputs, labels):
        self.model.fit(inputs, labels)
        
    def evaluate(self, inputs, labels):
        predictions = self.model.predict(inputs)
        accuracy = metrics.accuracy_score(labels, predictions)
        print("准确率是", accuracy)
    

def tradicional_methods():
    FE = FeatureEngineering()
    FE.process()
    
    model = TratitionalModel()
    data = FE.load_corpus(run_time.PATH_TOUTIAO_DATA_NGRAMS)
    print(data.keys())
    inputs, labels = data['inputs'], data['labels']
#     for line in inputs: print(sum(line))
    inputs = np.array(inputs)
    labels = list(map(lambda x: [x], labels))
    labels = np.array(labels)
    
    inputs, test_inputs, labels, test_labels = train_test_split(inputs, labels, test_size=0.05)
    print("input的形状是", inputs.shape, len(labels))
    print("训练")
    model.fit(inputs, labels)
    print("测试")
    model.evaluate(test_inputs, test_labels)
if __name__ == '__main__':
    tradicional_methods()
    
    
    