import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 
import albumentations as A


class LMDataset(Dataset): 
    def __init__(self, csv, aug=None, normalization='simple', is_test=False): 
        self.labels = csv.label.values
        self.csv = csv.path.values
        self.aug = aug
        self.normalization = normalization
        self.is_test = is_test

    def __getitem__(self, index):
        img_path = self.csv[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug:
            img = self.augment(img)
        img = img.astype(np.float32)
        
        if self.normalization:
            img = self.normalize_img(img)

        tensor = self.to_torch_tensor(img)
        if self.is_test:
            feature_dict = {'idx':torch.tensor(index).long(),
                            'input':tensor}
        else:
            target = torch.tensor(self.labels[index])
            feature_dict = {'idx':torch.tensor(index).long(),
                            'input':tensor,
                            'target':target.float().long()}
        return feature_dict

    def __len__(self): 
        return len(self.csv)

    def augment(self,img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def normalize_img(self,img):
        if self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img = img-mean
            img = img*np.reciprocal(std, dtype=np.float32)
        else:
            pass
        return img
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img