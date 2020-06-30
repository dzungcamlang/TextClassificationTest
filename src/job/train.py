'''
Created on 2020��4��6��

@author: Administrator
'''
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from models.traditional_algorithms import tradicional_methods
from config import run_time
import numpy as np
from sklearn.model_selection import train_test_split

def train_each():
    tradicional_methods()
    
if __name__ == '__main__':
    train_each()