import os
from os import path, walk
from shutil import copyfile
import random
import string

import sklearn.model_selection

# from data_management import data_utils
import data_utils


def distribute_annotated_images(data_root, target_root, class_label, num_splits=3, split_distributions = [0.7, 0.2, 0.1]):
    assert(num_splits in (2,3))
    splits = ['train','val', 'test']
    all_img_paths = get_all_img_paths(data_root)

    for i,_ in enumerate(splits):
        create_split_folder(target_root, splits[i])
    
    images_per_split = determine_img_outputs(all_img_paths, split_distributions)
    relocate_images(images_per_split, target_root, class_label)

def create_split_folder(target_dir, split, overwrite=False):
    target_path = path.join(target_dir,split)
    if overwrite == True:
        os.remove(target_path)
    data_utils.create_dir_if_not_exist(target_path)

def get_all_img_paths(data_root):
    image_paths = []
    for root, _, files in walk(data_root):
        for img in files:
            image_paths.append(path.join(root,img))
    return(image_paths)

def determine_img_outputs(image_paths, split_distributions, splits=3):
    if splits == 2:
        return(sklearn.model_selection.train_test_split(image_paths, test_size=split_distributions[1], random_state=1))
    elif splits ==3:
        val_test_ratio = split_distributions[2] / split_distributions[1]
        train, val = sklearn.model_selection.train_test_split(image_paths, test_size=split_distributions[1], random_state=1)
        val, test = sklearn.model_selection.train_test_split(val, test_size=val_test_ratio, random_state=1) # Split val into val, test
        return([train, val, test])        

def relocate_images(images_per_split, target_dir, class_label):
    splits = ['train','val', 'test']
    for i,_ in enumerate(images_per_split):
        data_utils.create_dir_if_not_exist(path.join(target_dir, splits[i], class_label))

    for i,_ in enumerate(images_per_split):
        for image_source_path in images_per_split[i]:
            img_type = '.'+path.basename(image_source_path).split('.')[-1]
            file_name_out = data_utils.generate_random_filename(length=10) + img_type
            image_target_path = path.join(target_dir, splits[i], class_label, file_name_out)
            copyfile(image_source_path, image_target_path)