# encoding: utf-8
import torch
import torchvision.transforms as T
from PIL import Image

from models import Encoder

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 载入CNN模型
encoder = Encoder()
encoder.to(device)
encoder.eval()

# 数据增强
preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

# 测试CNN
if __name__ == '__main__':
    img_path = 'datasets/Flicker8k_Dataset/19212715_20476497a3.jpg'
    ori_img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(ori_img)
    img_tensor = img_tensor.unsqueeze(0)
    encoder_out = encoder(img_tensor)
    print(encoder_out.shape)
    print(encoder_out)

