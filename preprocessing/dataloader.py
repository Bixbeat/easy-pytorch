"""
Sources:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/camvid_loader.py
    https://github.com/bodokaiser/piwise/blob/master/piwise/dataset.py
    https://github.com/bfortuner/pytorch_tiramisu/blob/master/camvid_dataset.py    
"""

import os, os.path
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import torch
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.datasets.folder import is_image_file, default_loader

def decode_image(tensor, mean, sdev):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(sdev)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def get_normalize_params(all_image_filepaths):
    band_mean = [[],[],[]]
    band_std = [[],[],[]]
    for i, file in enumerate(all_image_filepaths):
        current_img = misc.imread(file)
        
        for band in range(3):
            band_mean[band] += [np.mean(current_img[:,:,band])]
            band_std[band] += [np.std(current_img[:,:,band])]
                
    for i in range(len(band_mean)):
        band_mean[i] = np.mean(band_mean[i])
        band_std[i] = np.mean(band_std[i])
        
    return band_mean, band_std

def get_images(root_filepath, sort = True):
    image_paths = []
    for image in os.listdir(root_filepath):
        # if is_image_file(image):
        full_img_path = os.path.join(root_filepath, image)
        image_paths.append(full_img_path)
    if sort == True:
        return sorted(image_paths)
    return image_paths

class ImageDataset(data.Dataset):
    """TODO: Reconsider iterator to allow cropping"""
    def __init__(self, root_path, classes, class_weights, split='train', input_transform=None, target_transform=None):
        self.images_root = os.path.join(root_path, 'images')
        self.labels_root = os.path.join(root_path, 'labels')
        assert split in ('train', 'val', 'test')
        self.split = split
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.n_classes = classes
        self.class_weights = class_weights
        
        self.imgs = get_images(os.path.join(self.images_root, self.split))
        self.labels = get_images(os.path.join(self.labels_root, self.split))

    def set_classes(self, classnames, weights):
        self.classnames = classnames
        self.class_weight = weights
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        lbl_path = self.labels[index]

        img = misc.imread(img_path)
        img = np.array(img, dtype=np.int32)

        lbl = Image.open(lbl_path) #.convert('1')
        lbl_np = (np.array(lbl, dtype=np.uint8)
        # lbl_torch = torch.from_numpy(lbl_np).long()

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            lbl_torch = self.target_transform(lbl_torch)

        return img, lbl_torch, img_path

    def __len__(self):
        return len(self.imgs)

def keep_mixed_class_labels(img_paths, lbl_paths):
    """For any combination of images and labels,
    Keeps only the labels that contain >1 class"""
    corrected_img_paths = []
    corrected_lbl_paths = []
    
    for i,label in enumerate(lbl_paths):
        lbl = np.array(Image.open(label))
        if np.max(lbl) != np.min(lbl):
            corrected_img_paths.append(img_paths[i])
            corrected_lbl_paths.append(lbl_paths[i])
    
    return corrected_img_paths, corrected_lbl_paths
        
        
if __name__ == "__main__":
    # Get normalize parameters
    imgs = get_images("/home/anteagroup/Documents/deeplearning/code/bag_project/data/images/train/")
    normalize_params = get_normalize_params(imgs)
    
    img_directory = "/home/anteagroup/Documents/deeplearning/code/bag_project/data/"
    imgdata = ImageDataset(split = "test", root_path = img_directory)
    
    example_inputs, example_targets = next(iter(imgdata))