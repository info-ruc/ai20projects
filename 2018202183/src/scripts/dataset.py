
import h5py
import json
import os
import torch
from torch.utils.data import Dataset



class CaptionDataset(Dataset):

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: 数据文件路径
        :param data_name: 数据名
        :param split: 分为'TRAIN', 'VAL','TEST'
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # 打开hdf5文件
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # 每张照片的字幕
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        self.transform = transform

        # datapoints数量
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i // self.cpi].transpose((1, 2, 0))
        # img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size