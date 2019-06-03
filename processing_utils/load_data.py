#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:50:14 2017

@author: anteagroup
"""
import sys
import os
import csv
from glob import glob
import collections
from future.utils import raise_from

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

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, img_dir, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.img_dir = img_dir
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        
        if self.transform:
            img = self.transform(img)

        return [img, annot]

    def load_image(self, image_index):
        img = Image.open(os.path.join(self.img_dir, self.image_names[image_index]))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)