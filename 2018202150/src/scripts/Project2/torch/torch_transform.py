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
from torch_dataset import mydataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_dataset = mydataset(train=True,transform=data_transforms['train'])
valid_dataset = mydataset(train=False,transform=data_transforms['valid'])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True,num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=4, shuffle=True,num_workers=0)

dataset_sizes = {x: len(os.listdir(os.path.join('E:\\AutoDrive\\data',x))) for x in ['train', 'valid']}
dataloaders = {'train':train_loader,'valid':valid_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")