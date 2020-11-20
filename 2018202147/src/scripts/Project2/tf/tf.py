from tf_inference import inference
from tf_preprocess import preprocess
from tf_train import train
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os

if __name__ == "__main__":
    train_file='E:\\AutoDrive\\train.p'
    valid_file='E:\\AutoDrive\\valid.p'
    test_file = 'E:\\AutoDrive\\test.p'
    x_train, x_valid, x_test,y_train,y_valid,y_test = preprocess(train_file, valid_file, test_file)
    train(x_train, y_train)
    
