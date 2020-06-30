'''
Created on 2020年4月22日

@author: Administrator
'''
import pickle

def save_model(model, target_file):
    pickle.dump(model, open(target_file, 'wb'))
    
def load_model(target_file):
    return pickle.load(open(target_file, 'rb'))