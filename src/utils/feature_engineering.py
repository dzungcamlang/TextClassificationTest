'''
Created on 2020年4月6日

@author: Administrator
'''

class BigramFeature():
    
    def __init__(self):
        self.tf_idf = {}
        self.tf = {}
        self.df = {}
    
    def get_bigram_freature(self, text_list):
        bigrams_list = []
        tf_list = []
        for text in text_list:
            bigrams = self.get_bigram(text)
            tf_in_text = {}
            for bigram in set(bigrams):
                self.df[bigram] = self.df.get(bigram, 0) + 1
            for bigram in bigrams:
                self.tf[bigram] = self.tf.get(bigram, 0) + 1

        for bigram in self.df:
            self.tf_idf = self.tf[bigram]/self.df[bigram]
        
        start_index, end_index = int(len(self.tf_idf)*0.05), int(len(self.tf_idf)*0.5) 
        left_bigrams = sorted(self.tf_idf.items(), key=lambda x: x[1], reverse=True)[start_index: end_index]
    
    def get_bigram(self, text):
        bigrams = []
        for i in range(len(text) - 1):
            bigrams.append(text[i: i + 2])
        return bigrams
        