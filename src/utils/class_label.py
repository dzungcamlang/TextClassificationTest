'''
Created on 2020年4月6日

@author: Administrator
'''

class ClassLabelProcess():
    
    def __init__(self, class_name_set):
        self.class_name_label_map = {}
        self.class_name_onehot_label = {}
        self.generate_class_label(class_name_set)
    
    def generate_class_label(self, class_name_set):
        class_names = list(class_name_set)
        for i in range(len(class_names)):
            self.class_name_label_map[class_names[i]] = i
            
            one_hot_label = [0 for _ in range(len(class_names))]
            one_hot_label[i] = 1
            self.class_name_onehot_label[class_names[i]] = one_hot_label
    def get_label(self, class_name):
        return self.class_name_label_map[class_name], self.class_name_onehot_label[class_name]
        
            