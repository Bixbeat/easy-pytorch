"""
Sources:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/camvid_loader.py
    https://github.com/bodokaiser/piwise/blob/master/piwise/dataset.py
    https://github.com/bfortuner/pytorch_tiramisu/blob/master/camvid_dataset.py    
"""
import os, os.path
import numpy as np
import shutil

from PIL import Image, ImageOps

def get_normalize_params(all_image_filepaths, num_bands):
    """For a set of image filepaths, returns the mean
    and stdev of all bands of all images in the set
    """
    band_mean = [[] for i in range(num_bands)]
    band_stdev = [[] for i in range(num_bands)]

    for i, file in enumerate(all_image_filepaths):
        current_img = Image.open(file)
        if num_bands == 3 and current_img.mode in ['L', 'P']: #Stopgap for B&W images
            bw_image = current_img
            current_img = Image.new("RGB", current_img.size)
            current_img.paste(bw_image)              
        current_img = np.asarray(current_img)
    
        for band in range(num_bands):
            band_mean[band].append(np.mean(current_img[:,:,band]))
            band_stdev[band].append(np.std(current_img[:,:,band]))
                
    for i,_ in enumerate(band_mean):
        band_mean[i] = np.mean(band_mean[i])
        band_stdev[i] = np.mean(band_stdev[i])

    return {'means': band_mean, 'sdevs':band_stdev}

def get_images(root_filepath, sort=True):
    """For a given path, returns a (sorted) list containing all
    files."""
    image_paths = []
    for folder, _, imgs in os.walk(root_filepath):
        for image_path in imgs:
            image_paths.append(os.path.join(folder, image_path))
    if sort is True:
        image_paths = sorted(image_paths)
    return image_paths

def keep_mixed_class_labels(img_paths, lbl_paths):
    """For any combination of image and label paths with
    identical filenames, Keeps only the labels that contain >1 class"""
    corrected_img_paths = []
    corrected_lbl_paths = []

    for i, label in enumerate(lbl_paths):
        lbl = np.asarray(Image.open(label))
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
        shutil.copyfile(missing_lbl, root_dir + lbl_path + '/' + file)
    
    for file_path in all_lbls:
        file = os.path.basename(file_path)
        missing_img = root_dir + src_data_root + "tiles/" + file
        shutil.copyfile(missing_img, root_dir + img_path + '/' + file)

def img_to_thumbnail(pil_img, width=400):
    # Credit: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    wpercent = (width/float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1])*float(wpercent)))
    return(pil_img.resize((width,hsize), Image.ANTIALIAS))

def delete_equal_images_in_same_folder(root_dir):
    total_deleted = 0
    for root, _, files in os.walk(root_dir):
        for image1 in files:
            img_path_1 = os.path.join(root, image1)
            for image2 in files:
                if is_image(image1) and is_image(image2):
                    img_path_2 = os.path.join(root, image2)                    
                    remove_path_from_list = del_image_if_equal(img_path_1, img_path_2)
                    if remove_path_from_list == True:
                        files.remove(image2)
                        total_deleted +=1
    print(f'total images deleted: {total_deleted}')
    
def delete_equal_images_from_root(root_dir):
    """For every image, checks every other image
    in that root directory for equivalence.
    If equivalent, it removes the equivalent image
    and retains the original image."""
    # Welcome to control structure hell
    total_deleted = 0
    remaining_subdirs = [x[0] for x in os.walk(root_dir)]
    for img1_root, _, img1_files in os.walk(root_dir):
        for image1 in img1_files:
            if len(remaining_subdirs) >= 1:
                for folder in remaining_subdirs:
                    for img2_root, _, img2_files in os.walk(folder):
                        for image2 in img2_files:
                            if is_image(image1) and is_image(image2):
                                img_path_2 = os.path.join(img2_root, image2)
                                img_path_1 = os.path.join(img1_root, image1)
                                remove_path_from_list = del_image_if_equal(img_path_1, img_path_2)
                                if remove_path_from_list == True:
                                    if image2 in img1_files:                                
                                        img1_files.remove(image2)
                                    total_deleted +=1
        remaining_subdirs.remove(img1_root)
    print(f'total images deleted: {total_deleted}')                            

def is_image(file_path):
    _, file_extension = os.path.splitext(file_path)
    return (file_extension.lower() in ['.png','.jpg','.jpeg','.tif','.tiff','.gif'])

def zero_pad_img(pil_img, target_size):
    fitted_img = ImageOps.fit(pil_img, (target_size, target_size))
    return(fitted_img)    

def del_image_if_equal(img_path_1, img_path_2):
    img_1 = Image.open(img_path_1)
    img_2 = Image.open(img_path_2)
    remove_path = False
    if img_1 == img_2 and img_path_1 != img_path_2:
        print(f'{img_path_1} is equal to {img_path_2} \n')
        os.remove(img_path_2)
        remove_path = True
    img_1.close()
    img_2.close()
    return(remove_path)

def image_to_dataloader_folders(dataloader_root, img_class, img_split, img_path, output_img_width='original', crop_bottom=0):
    img_name = os.path.basename(img_path)
    target_img_path = os.path.join(dataloader_root, img_split, str(img_class), img_name)
    if output_img_width != 'original':
        try:
            img_to_resize = Image.open(img_path)
            # TODO: Make this function less convoluted
            if crop_bottom > 0:
                img_to_resize = crop_bottom_and_sides(img_to_resize, crop_bottom)
            img_to_equal_resize = ImageOps.fit(img_to_resize, (output_img_width, output_img_width))
            img_to_equal_resize.save(target_img_path)
        except FileNotFoundError as e:
            print(f'file not found: {e}\n')        
    else:
        try:
            shutil.copyfile(img_path, target_img_path)
        except shutil.Error as e:
            print(f'Cannot copy file: {e}\n')

def crop_bottom_and_sides(pil_img, bottom_ratio):
    """Esoteric method to deal with the ego vehicle being in the bottom 
    of many frames in the BDD dataset. Crops out the full ratio from the
    bottom and half from the left & right to maintain an equal aspect ratio"""
    array_to_resize = np.asarray(pil_img)
    height, width, _ = np.shape(array_to_resize)
    new_height = int(height*(1-bottom_ratio))-1 # Offset index
    width_slice = int(width*(bottom_ratio/2)) # Slice half of left & right
    cropped_array = array_to_resize[:new_height, width_slice:(width-width_slice),:,]
    return Image.fromarray(cropped_array)

        # bottom_cropped_img = img_to_resize[;,;,;,]
        # img_to_equal_resize = ImageOps.fit(resized_img, (output_img_width, output_img_width))