import torch
import torchvision as tv
from PIL import Image
from torch.autograd import Variable
import os
import csv

def test():
    result = []
    classes = get_classes()
    for i in range(5):
        result.append(classes[i + 1])
    return result

def get_classes():
    classes = {}
    with open("static/model/classes.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            classes[int(row[0])] = row[1]
    return classes

def classify(data_path):
    model = torch.load("static/model/leafsnap_model2.pth", map_location = lambda storage, loc: storage)
    model = model.module
    normalize = tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = tv.transforms.Compose([tv.transforms.ToTensor(), normalize])
    classes = get_classes()
    img = Image.open(data_path)
    img = data_transforms(img).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        ans = torch.topk(output.data, 5)

    classes_num = ans.indices.cpu().numpy()[0]
    print(classes_num)
    result = []
    for i in range(len(classes_num)):
        result.append(classes[classes_num[i]])
    
    return result

