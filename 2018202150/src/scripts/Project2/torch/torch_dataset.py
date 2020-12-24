import os
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import torch
import time
import os
import PIL
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# define a class for dataloader and data iter.
class mydataset(torch.utils.data.Dataset):
    def __init__(self, root='E:\\AutoDrive\\data\\train', train=True, transform = None, target_transform=None):
        super(mydataset, self).__init__()
        self.train = train
        self.transform = transform
        self.targets=[2,3,21,22,24,31,42,52]
        self.target_transform = target_transform
        if self.train:
            img_dir='E:\\AutoDrive\\data\\train'
        else:
            img_dir='E:\\AutoDrive\\data\\valid'
        self.img_names = os.listdir(img_dir)
        self.img_dir=img_dir

    # overload [] operator for get image and label by index
    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = int(img_name.split('_')[0])
        label=self.targets.index(label)
        img = PIL.Image.open(os.path.join(self.img_dir,img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    # overload len() operator to get the size of dataset
    def __len__(self):
        return len(os.listdir(self.img_dir))