#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:50:14 2017

@author: anteagroup
"""
from . import image_manipulations as i_manips

import os
from PIL import Image

from . import joint_transforms as j_trans

import numpy as np
import torch
from torch.utils.data import Dataset

class SemSegImageData(Dataset):
    """Wrapper for loading image segmentation"""
    def __init__(self, root_path, split='train', input_transform=None, target_transform=None, joint_trans=None):
        self.images_root = os.path.join(root_path, 'images')
        self.labels_root = os.path.join(root_path, 'labels')
        
        assert split in ('train', 'val', 'test')
        self.split = split
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.joint_transforms = joint_trans
        
        self.imgs = i_manips.get_images(os.path.join(self.images_root, self.split))
        self.labels = i_manips.get_images(os.path.join(self.labels_root, self.split))
    
    def __getitem__(self, index):
        """For a given index, returns an image and label pair with the
        specified transformations
        TODO: fix separate loading of labels"""
        img_path = self.imgs[index]
        lbl_path = self.labels[index]

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        
        if 'hflip' in self.joint_transforms:
            img, lbl = j_trans.joint_horizontal_flip(img, lbl)
        if 'vflip' in self.joint_transforms:
            img, lbl = j_trans.joint_vertical_flip(img, lbl)
        # img = np.array(img, dtype=np.int32)
        # lbl_np = (np.array(lbl, dtype=np.uint8)) # [:,:,2] Why was this here?
        # target_lbl = torch.from_numpy(lbl_np).long()

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            target_lbl = self.target_transform(lbl)

        return img, target_lbl, img_path

    def __len__(self):
        return len(self.imgs)
