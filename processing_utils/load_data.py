#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:50:14 2017

@author: anteagroup
"""
import os
from glob import glob
import collections

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

from . import joint_transforms as j_trans
from . import image_manipulations as i_manips

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

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)

        return img, lbl, img_path

class LoadVOCStyleData(Dataset):
    """Dataloader for labels in the style of Pascal VOC XMLs
    """
    def __init__(self, root_dir, split='train', input_transform=None):
        assert split in ('train', 'val', 'test')
        self.target_dir = os.path.join(root_dir, split)
        self.annotations = glob(f'{self.target_dir}/*.xml')

        self.split = split
        self.input_transform = input_transform
    
    def __getitem__(self, index):
        """For a given index, returns an image and label pair with the
        specified transformations"""
        target = self._parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        image = Image.open(os.path.join(self.target_dir, target['annotation']['filename'])).convert('RGB')

        if self.input_transform is not None:
            image = self.input_transform(image)    

        return image, target 

    def _parse_voc_xml(self, node):
        """Source: https://pytorch.org/docs/master/_modules/torchvision/datasets/voc.html"""
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self._parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __len__(self):
        return len(self.annotations)
