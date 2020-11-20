import os
import cv2

img_names=os.listdir('E:\\AutoDrive\\data\\crop')
count={}
for img_name in img_names:
    img_path='E:\\AutoDrive\\data\\crop'+img_name
    label = int(img_name.split('_')[0])
    # only count interesting labels
    if(label not in count.keys()):
        count[label]=1
    else:
        count[label] += 1

# split train and valid set
targets=[2,3,21,22,24,31,42,52] # interesting labels
count_train={}
count_valid={}
for target in targets:
    count_train[target]=0
    count_valid[target]=0
for img_name in img_names:
    img_path='E:\\AutoDrive\\data\\crop\\'+img_name
    label=int(img_name.split('_')[0])
    if label not in targets:
        continue
    no=int(img_name.split('.')[0].split('_')[-1])
    if no >= 0.7 * count[label]:
        # divide into validation set
        count_valid[label]+=1
        img=cv2.imread(img_path)
        cv2.imwrite('E:\\AutoDrive\\data\\valid\\'+img_name,img)
    else:
        # divide into train set
        count_train[label]+=1
        img=cv2.imread(img_path)
        cv2.imwrite('E:\\AutoDrive\\data\\train\\'+img_name,img)   
        
