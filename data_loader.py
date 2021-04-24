import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
# torch.cuda.set_device(gpu_id)#使用GPU

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], words[1]))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        label_img = self.loader(label)
        if self.transform is not None:
            img = self.transform(img)
            label_img = self.transform(label_img)
        return img, label_img

    def __len__(self):
        return len(self.imgs)



train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])
text_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])


