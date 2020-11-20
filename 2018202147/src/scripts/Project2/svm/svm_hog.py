import os
import cv2
from openpyxl import Workbook 
import xlrd
import pandas as pd
import numpy as np
from skimage import feature as ft 
from sklearn.svm import SVC
import joblib
def hog_feature(img, resize=(64,64)):
    # extract hog feature from image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2" # define the method for normalization
    features = ft.hog(gray, orientations=bins, pixels_per_cell=cell_size, 
                        cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    return features

def extract_hog_features(img_dir, write_txt, resize=(64,64)):
    ##extract hog feature for every image in the img_dir and write the hog in to a text
    img_names = os.listdir(img_dir)
    if os.path.exists(write_txt):
        os.remove(write_txt)
    
    with open(write_txt, "a") as f:
        index = 0
        for img_name in img_names:
            img = cv2.imread(os.path.join(img_dir,img_name))
            features = hog_feature(img, resize)
            label= img_name.split("_")[0]
            row_data = img_name + "\t" + str(label) + "\t"
            
            # connect all the element in features into a line;
            for element in features:
                row_data = row_data + str(round(element,3)) + " "
            row_data = row_data + "\n"
            f.write(row_data)
            
            if index%100 == 0:
                print ("total image number = ", len(img_names), "current image number = ", index)
            index += 1

if __name__ == "__main__":
    extract_hog_features('E:\\AutoDrive\\data\\crop','E:\\AutoDrive\\data\\feature.txt')