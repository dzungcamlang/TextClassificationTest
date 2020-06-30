'''
Created on 2020年4月23日

@author: Administrator
'''
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import pickle
from config import run_time
from entity.corpus_processor import CorpusProcessor
import random

class CorpusLoader():
    
    def load_toutiao_cat_data(self, sample=None):
        print("加载原始数据")
        CL = CorpusProcessor()
        lines = CL.load(run_time.PATH_TOUTIAO_DATA)
        texts, labels, onehot_labels = [], [], []
        
        for line in lines:
            if sample!=None:
                if random.uniform(0, 1) > sample:
                    continue
            texts.append(line['text'])
            labels.append(line['class_label'])
            onehot_labels.append(line['class_label_one_hot_encoding'])
        return texts, labels, onehot_labels
    
    