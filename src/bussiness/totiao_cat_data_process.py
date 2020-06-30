# -*- coding:utf-8 -*-
'''
Created on 2020��4��6��

@author: Administrator
'''
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from entity.corpus_processor import CorpusProcessor
from utils.class_label import ClassLabelProcess
from config import run_time
import pandas as pd
import json
class ToutiaoProcessor(CorpusProcessor):
    
    def read_corpus(self):
        lines = list(open(run_time.PATH_TOUTIAO_ORI, 'r', encoding='utf8').readlines())
        self.data_json = []
        class_name_set = set()
        #第一阶段
        for line in lines:
            data = {"text": "", "class_name": "", "class_label": -1, "class_label_one_hot_encoding": []}
            slices = line.split("_!_")
            data['class_name'] = slices[2]
            class_name_set.add(data['class_name'])
            data['text'] = "".join(slices[3:])
            self.data_json.append(data)
        #获取类标签的独热编码
        class_label_processor = ClassLabelProcess(class_name_set)
        for data in self.data_json:
            data['class_label'], data['class_label_one_hot_encoding'] = \
                         class_label_processor.get_label(data['class_name'])
                         
    def save_as_tsv(self):
        lines = list(open(run_time.PATH_TOUTIAO_ORI, 'r', encoding='utf8').readlines())
        import random 
        random.shuffle(lines)
        max_len = 0
        with open("train.tsv", 'w', encoding='utf8') as f:
            for line in lines:
                slices = line.split("_!_")
                class_name = slices[2]
                text = "".join(slices[3:])
                if len(text)>max_len: max_len=len(text)
                new_line = class_name + "\t" + text
                new_line = new_line.replace('\0','')
                f.write(new_line)
        print('max_len', max_len)
        


if __name__ == '__main__':
    a = ToutiaoProcessor()
#     a.read_corpus()
#     a.save(run_time.PATH_TOUTIAO_DATA)
    a.save_as_tsv()
    
