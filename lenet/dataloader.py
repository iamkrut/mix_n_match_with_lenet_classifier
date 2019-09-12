import os
from os.path import isdir, exists, abspath, join
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class ChairsDataset(Dataset):
    
    def __init__(self, root_dir='chairs-data', train = True, transform=None):

        self.root_dir = abspath(root_dir) 
        self.train = train
        self.transform = transform

        if train:
            self.pos_dir = join(self.root_dir, 'train/positive')
            self.neg_dir = join(self.root_dir, 'train/negative')
        else:
            self.pos_dir = join(self.root_dir, 'test/positive')
            self.neg_dir = join(self.root_dir, 'test/negative')


        self.pos_files = os.listdir(self.pos_dir) 
        self.neg_files = os.listdir(self.neg_dir)

    def __len__(self):
        return len(self.pos_files) + len(self.neg_files)

    def positive_size(self):
        return len(self.pos_files)

    def negative_size(self):
        return len(self.neg_files)

    
    def __getitem__(self, idx):
        
        if idx < len(self.pos_files):
            img_name = join(self.pos_dir, self.pos_files[idx])
            image = Image.open(img_name)

            if self.transform:
                image = self.transform( image )

            img_label = 1
            sample = (image, img_label)

        else:
            img_name = join(self.neg_dir, self.neg_files[idx-len(self.pos_files)])
            image = Image.open(img_name)

            if self.transform:
                image = self.transform( image )

            img_label = 0
            sample = (image, img_label)

        return sample

        

        
        
