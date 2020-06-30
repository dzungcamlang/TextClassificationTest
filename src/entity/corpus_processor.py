# -*- coding:utf-8 -*-
'''
Created on 2020��4��6��

@author: Administrator
'''
import json

#所有语料处理器的基类
class CorpusProcessor():
    
    def __init__(self):
        self.data_json = []
    
    #按照所需的格式读取语料
    def read_corpus(self):
        pass
    
    #将语料处理为特定的格式
    def process(self):
        pass
    
    #存储语料
    def save(self, target_file):
        json.dump(self.data_json, open(target_file, 'w'))
        
    def load(self, target_file):
        return json.load(open(target_file, 'r'))
