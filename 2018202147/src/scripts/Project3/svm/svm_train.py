from svm_labels import Label
import os
import cv2
from openpyxl import Workbook 
import xlrd
import pandas as pd
import numpy as np
from skimage import feature as ft 
from sklearn.svm import SVC
import joblib
def load_hog_data(hog_txt):
    img_names = []
    labels = []
    hog_features = []
    with open(hog_txt, "r") as f:
        data = f.readlines()
        for row_data in data:
            row_data = row_data.rstrip()
            try:
                img_name, label, hog_str = row_data.split("\t")
            except ValueError:
                print(row_data.split("\t")[0])
                continue
            hog_feature = hog_str.split(" ")
            hog_feature = [float(hog) for hog in hog_feature]
            #print "hog feature length = ", len(hog_feature)
            img_names.append(img_name)
            labels.append(int(label))
            hog_features.append(hog_feature)
        for i in range(len(labels)):
            if labels[i] not in target:
                labels[i]=58
    return img_names, np.array(labels), np.array(hog_features)

def svm_train(hog_features, labels, save_path="./svm_model.pkl"):
    clf = SVC(C=10, tol=1e-3, probability = True)
    clf.fit(hog_features, labels)
    try:
        joblib.dump(clf, save_path)
        print("sucessfully saved")
    except:
        return clf
    print("finished.")
    
if __name__ == "__main__":
    img_names, labels, hog_train_features = load_hog_data("E:\\AutoDrive\\data\\feature.txt")
    clf=svm_train(hog_train_features, labels, "E:\\AutoDrive\\model\\svm_model2.pkl")