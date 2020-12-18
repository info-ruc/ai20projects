import os
import cv2
from openpyxl import Workbook 
import xlrd
import pandas as pd
import numpy as np
from skimage import feature as ft 
from sklearn.svm import SVC
import joblib

def center_crop(img_array, crop_size=-1, resize=-1, write_path=None):
    # resize and crop from the center
    rows = img_array.shape[0]
    cols = img_array.shape[1]

    if crop_size==-1 or crop_size>max(rows,cols):
        crop_size = min(rows, cols)
    # calculate the width and height
    row_s = max(int((rows-crop_size)/2), 0)
    row_e = min(row_s+crop_size, rows) 
    col_s = max(int((cols-crop_size)/2), 0)
    col_e = min(col_s+crop_size, cols)

    img_crop = img_array[row_s:row_e,col_s:col_e,]

    if resize>0:
        img_crop = cv2.resize(img_crop, (resize, resize))

    #write back the image to destination file
    if write_path is not None:
        cv2.imwrite(write_path, img_crop)
    return img_crop
    
count={}
def crop_img_dir(img_dir,  save_dir, crop_method = "center"):
##crop every img in the img_dir and resave them in the save_dir

    img_names = os.listdir(img_dir)
    img_names = [img_name for img_name in img_names if img_name.split(".")[-1]=="png"]
    index = 0
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        label = int(img_name.split('_')[0])
        # count by label
        if label not in count.keys():
            count[label]=1
        else:
            count[label] += 1
        # include label information in the new image name
        new_name=str(label)+'_'+str(count[label])+'.png'
        new_path=os.path.join(save_dir,new_name)
        if crop_method == "center":
            img_crop = center_crop(img, resize=640, write_path=new_path)
        if index%100 == 0:
            print ("total images number = ", len(img_names), "current image number = ", index)
        index += 1

if __name__ == "__main__":
    img_dir="E:\\AutoDrive\\data\\original"
    save_dir="E:\\AutoDrive\\data\\crop"
    crop_img_dir(img_dir,save_dir)