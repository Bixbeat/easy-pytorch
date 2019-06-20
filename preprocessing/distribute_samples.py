import os
import sys
from ntpath import basename
from shutil import copyfile

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

from data_management.data_utils import create_dir_if_not_exist
from processing_utils import image_manipulations as img_manips

def del_single_samples(img_folder, label_folder):
    """For a given image and label forder, keeps only
    files which have the same name"""
    # img_root = "/home/anteagroup/Documents/deeplearning/code/bag_project/data/images/test"
    # lbl_root = "/home/anteagroup/Documents/deeplearning/code/bag_project/data/labels/test"
    
    all_imgs = os.listdir(img_folder)
    all_lbls = os.listdir(label_folder)
    for img in os.listdir(all_imgs):
        if img not in all_lbls:
            os.remove(os.path.join(all_lbls,img))

class DataDistributor(object):
    """Takes a set of pre-existing labels and images, then distributes them into 
    Train/Val/Test folders randomly. Tiles can be approved by the user.
    
    TODO: Refactor the flow of filepaths and filenames through the system
    """
    def __init__(self, img_filepaths, label_filepaths, output_root):        
        self.output_root = output_root
        self.img_path = img_filepaths
        self.label_path = label_filepaths
    
        self.all_img_paths = None
        self.all_label_paths = None
        self.current_image = None
        self.current_label = None
        
        self.previous_image = None
        self.previous_label = None
        self.previous_image_name = None
        self.previous_label_name = None
        self.previous_image_split = None
        self.previous_label_split = None
        
        self.skip_to_index = None
        self.approve_inputs = None
        
    def __call__(self, seed=0, keep_only_mixed = True, approve_inputs=True, index = 0):      
        ### User parameters
        np.random.seed(seed)
        self.approve_inputs = approve_inputs
        self.skip_to_index = index

        self._get_sorted_images(self.img_path, self.label_path, keep_only_mixed)

        self._assert_equal_length()        
        self.distribute_data()

    def _get_sorted_images(self, img_root, lbl_root, keep_only_mixed=False):
        """For given images and label filepaths,
        stores a list of all filepaths in those folders.
        Keep_only_mixed returns only filepaths where the file contains >1 class
        """
        self.all_img_paths = img_manips.get_images(self.img_path, sort = True)
        self.all_label_paths = img_manips.get_images(self.label_path, sort = True)    
        
        if keep_only_mixed:
            imgs, lbls = img_manips.keep_mixed_class_labels(self.all_img_paths, self.all_label_paths)
            self.all_img_paths = imgs
            self.all_label_paths = lbls            
            
    def _assert_equal_length(self):
        if len(self.all_img_paths) != len(self.all_label_paths):
            raise ValueError("Tiles and images not of equal length")
        
    def distribute_data(self):       
        i = 0
        if self.skip_to_index: i += self.skip_to_index          
        
        while i < len(self.all_img_paths):
            self.current_image = self.all_img_paths[i]
            self.current_label = self.all_label_paths[i]
            
            self.current_image_name = basename(self.current_image)
            self.current_label_name = basename(self.current_label)
            
            print("\nCurrent index ", i, "of ", len(self.all_img_paths))
            print("\n", self.current_image_name)
            if self.approve_inputs is not False: 
                self._display_images()
            self._distribute_pairs()
            
            i += 1
        
    def _display_images(self):
        """Displays an image and a label side-by-side
        TODO: refactor"""
        img = misc.imread(self.current_image)
        label = misc.imread(self.current_label)    
        
        _, (imgplot, lblplot) = plt.subplots(1,2, figsize=(12, 6))
        imgplot.imshow(img, aspect='auto')
        imgplot.grid(color='r', linestyle='dashed', alpha=0.75)
        
        lblplot.imshow(label, aspect='auto')
        lblplot.grid(color='r', linestyle='dashed', alpha=0.75)
        
        plt.show(block=False) # To force image render while user input is also in the pipeline

    def _distribute_pairs(self):
        if self.approve_inputs == True:
            user_cmd = None
            proceed = False
            while proceed == False:
                user_cmd = input("Type 1 to approve image, 0 to skip, rp to remove previous, sp to save previous, s to stop: ")
                if user_cmd in ["1","0","sp","rp","s"]:
                    proceed = self._handle_input(user_cmd)
        else:
            output_location = self._determine_split()
            self._store_split(output_location, self.current_image_name, self.current_label_name)

    def _handle_input(self, user_input):   
        if user_input == "1":
            output_location = self._determine_split()
            self._store_split(output_location, self.current_image_name, self.current_label_name)
            self._store_previous(output_location)
            
            proceed = True
            
        elif user_input == "sp":
            output_location = {"img_path":self.previous_label_split,"lbl_path":self.previous_label_split}
            self._store_split(output_location,self.previous_image, self.previous_label)                
                
            proceed = False
            
        elif user_input == "rp":
            try:
                previous_img_path = os.path.join(self.previous_image_split, self.previous_image_name)
                previous_lbl_path = os.path.join(self.previous_label_split, self.previous_label_name)
                os.remove(previous_img_path)
                os.remove(previous_lbl_path)
            except:
                print("\nFile not found")
                
            proceed = False
            
        elif user_input == "0":
            proceed = True
            
        elif user_input == "s":
            sys.exit()
             
        return proceed
    
    def _determine_split(self):
        """Returns the filepath of the split when an image is approved using random chance
        """
        folder_id = np.random.randint(1,101)
        
        if folder_id in range(0, 5):
            output_img_path = os.path.join(self.output_root, "images/val")
            output_label_path = os.path.join(self.output_root, "labels/val")
            
        elif folder_id in range(6, 40):
            output_img_path = os.path.join(self.output_root, "images/test")
            output_label_path = os.path.join(self.output_root, "labels/test")
            
        else:
            output_img_path = os.path.join(self.output_root, "images/train")
            output_label_path = os.path.join(self.output_root, "labels/train")
        
        create_dir_if_not_exist(output_img_path)
        create_dir_if_not_exist(output_label_path)

        return {"img_path": output_img_path, "lbl_path": output_label_path}
    
    def _store_split(self, output_location, image_path, label_path):
        img_path = os.path.join(output_location["img_path"], basename(image_path))
        lbl_path = os.path.join(output_location["lbl_path"], basename(label_path))
        
        copyfile(self.current_image, img_path)
        copyfile(self.current_label, lbl_path)    
    
    def _store_previous(self, output_location):
        self.previous_image_split = output_location["img_path"]
        self.previous_label_split = output_location["lbl_path"]
        
        self.previous_image = self.current_image
        self.previous_label = self.current_label
        
        self.previous_image_name = basename(self.previous_image)
        self.previous_label_name = basename(self.previous_label)      

def determine_split(split_probs=[75,15]):
    folder_id = np.random.randint(1,101)
    
    if folder_id in range(0, split_probs[0]):
        split = 1
        
    elif folder_id in range(split_probs[0], split_probs[0] + split_probs[1]):
        split = 2
        
    else:
        split = 3

    return split

if __name__ == "__main__":
    root_path = "/home/anteagroup/Documents/deeplearning/code/bag_project_p2/data"
    img_path = os.path.join(root_path, "rasters/out/tiles/nir/")
    label_path = os.path.join(root_path, "rasters/out/labels/")

    distribute = DataDistributor(img_path, label_path, root_path)
    distribute(approve_inputs = False, index=0, keep_only_mixed = False)
        