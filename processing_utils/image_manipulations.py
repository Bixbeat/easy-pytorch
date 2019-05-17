"""
Sources:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/camvid_loader.py
    https://github.com/bodokaiser/piwise/blob/master/piwise/dataset.py
    https://github.com/bfortuner/pytorch_tiramisu/blob/master/camvid_dataset.py    
"""
import glob

import os, os.path
import numpy as np
import scipy.misc as misc
from shutil import copyfile
from PIL import Image

def get_normalize_params(all_image_filepaths, num_bands):
    """For a set of image filepaths, returns the mean
    and stdev of all bands of all images in the set
    TODO: Expand beyond 3 bands"""
    band_mean = [[] for i in range(num_bands)]
    band_stdev = [[] for i in range(num_bands)]
    for i, fname in enumerate(all_image_filepaths):
        current_img = np.asarray(Image.open(fname))
        
        if not num_bands == 1:
            for band in range(num_bands):
                band_mean[band].append(np.mean(current_img[:,:,band]))
                band_stdev[band].append(np.std(current_img[:,:,band]))
        else:
            band_mean[0].append(np.mean(current_img))
            band_stdev[0].append(np.std(current_img))
                
    for i,_ in enumerate(band_mean):
        band_mean[i] = float(np.mean(band_mean[i]))
        band_stdev[i] = float(np.mean(band_stdev[i]))
        
    return {"means": band_mean, "sdevs":band_stdev}

def get_images(root_filepath, sort=True):
    """For a given path, returns a (sorted) list containing all
    files."""
    image_paths = []
    file_types = [ ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif"]

    for ftype in file_types:
        image_paths.extend(glob.iglob(root_filepath+'**/*'+ftype, recursive=True))
    if sort:
        image_paths = sorted(image_paths)

    return image_paths

def keep_mixed_class_labels(img_paths, lbl_paths):
    """For any combination of image and label paths with
    identical filenames, Keeps only the labels that contain >1 class"""
    corrected_img_paths = []
    corrected_lbl_paths = []
    
    for i,label in enumerate(lbl_paths):
        lbl = np.array(Image.open(label))
        if np.max(lbl) != np.min(lbl):
            corrected_img_paths.append(img_paths[i])
            corrected_lbl_paths.append(lbl_paths[i])
    return corrected_img_paths, corrected_lbl_paths

def sync_img_and_lbls(root_dir, src_data_root, img_path, lbl_path):
    """Hotfix for accidentally deleted data.
    For any combination of image and label paths with
    identical filenames, copies files from source data path
    to ensure synchronous imgs & labels
    TODO: fix repetitive code, add conditional"""
    all_imgs = get_images(root_dir+img_path)
    all_lbls = get_images(root_dir+lbl_path)
    
    for file_path in all_imgs:
        file = os.path.basename(file_path)
        missing_lbl = root_dir + src_data_root + "labels/" + file
        copyfile(missing_lbl, root_dir + lbl_path + '/' + file)
    
    for file_path in all_lbls:
        file = os.path.basename(file_path)
        missing_img = root_dir + src_data_root + "tiles/" + file
        copyfile(missing_img, root_dir + img_path + '/' + file)