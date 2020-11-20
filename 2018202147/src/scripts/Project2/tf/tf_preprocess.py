import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os

def preprocess(train_file, valid_file, test_file):
    with open(train_file,mode='rb') as f:
        train=pickle.load(f)
    with open(test_file,mode='rb') as f:
        test=pickle.load(f)
    with open(valid_file,mode='rb') as f:
        valid=pickle.load(f)
    x_train,y_train=train['features'],train['labels']
    x_valid,y_valid=valid['features'],valid['labels']
    x_test,y_test=test['features'],test['labels']
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    image_shape = x_train.shape[1:]
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    # number of training examples =34799
    # number of testing examples =12630
    # image data shape=(32,32,3)

    # conver rgb image to gray image
    x_train_rgb = x_train
    x_train_gry = np.sum(x_train/3, axis=3, keepdims=True)
    x_valid_rgb=x_valid
    x_valid_gry=np.sum(x_valid/3,axis=3,keepdims=True)
    x_test_rgb = x_test
    x_test_gry = np.sum(x_test/3, axis=3, keepdims=True)
    print('RGB shape:', x_train_rgb.shape)
    print('Grayscale shape:', x_train_gry.shape)
    # grb shape :(34799,32,32,3)
    # gray shape:(34799,32,32,1)

    #normalization
    x_train = (x_train_gry - 128.)/128. 
    x_valid=(x_valid_gry-128.)/128.
    x_test = (x_test_gry - 128.)/128.

    return x_train,x_valid,x_test,y_train,y_valid,y_test

