import string
import random
import os
import shutil
import operator
import csv

import sqlite3

from data_management.image_manipulations import is_image

class ImgDatabaseHandler():
    """IMPORTANT NOTE: Don't use this code in production without cleaning the input on the table name!
    Sets up a database connection with functions to save images.
    Args:
        db_root (string): Path to the database.
    """
    def __init__(self, db_root):
        self.db = sqlite3.connect(db_root)
        self.cursor = self.db.cursor()
        self.previous_img_path = None
        self.previous_geo = None
        self.previous_time = None

    def create_img_table(self, name):
        try:
            self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS
                                    {name}(img_id VARCHAR PRIMARY KEY, lat REAL, long REAL, datetime VARCHAR, class STRING)''')
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise e

    def store_image_details(self, target_table, image_class, img_path, location, time, autocommit=True):
        try:
            self.db.execute(f'INSERT INTO {target_table} VALUES (?,?,?,?,?)', [img_path, location[0], location[1], time, image_class])
            if autocommit:
                self.db.commit()
        except sqlite3.IntegrityError as e:
            print(f'Cannot store image details: {e}')
    
    def remove_record(self, target_table, img_path):
        try:
            self.db.execute(f'DELETE FROM {target_table} WHERE img_id=(?)', [img_path])
            self.db.commit()
        except sqlite3.IntegrityError as e:
            print(f'Cannot delete record: {e}')
    
    def get_all_records(self, table):
        try:
            return([record for record in self.cursor.execute(f'SELECT * FROM {table}')])
        except sqlite3.IntegrityError as e:
            print(f'Cannot load records: {e}')
    
    def calculate_splits(self, table, field_name, split_probabilities, commit_interval=50):
        records = self.get_all_records(table)
        for i, record in enumerate(records):
            split = determine_split(split_probabilities)
            try:
                self.db.execute(f'UPDATE {table} SET {field_name}=(?) WHERE img_id=(?)', [split, record[0]])
                if i % commit_interval == 0 or i == len(records):
                    self.db.commit()
            except sqlite3.IntegrityError as e:
                print(f'Cannot load records: {e}')
                
    def add_field(self, table, field_name):
        try:
            self.db.execute(f'ALTER TABLE {table} ADD COLUMN {field_name} INTEGER;')
        except sqlite3.IntegrityError as e:
            self.db.rollback()
            print(f'Cannot add split field: {e}')
    
    def del_missing_imgs(self, table, path_table_index, path_root):
        try:
            all_entries = self.get_all_records(table)
        except sqlite3.IntegrityError as e:
            self.db.rollback()
            print(f'Cannot select entries: {e}')
        for entry in all_entries:
            img_path = os.path.join(path_root, entry[path_table_index])
            if not os.path.isfile(img_path):
                self.db.execute(f'DELETE FROM {table} WHERE img_id=(?)', [entry[path_table_index]])
                print(f"Deleted nonexisting path from db: {img_path}")
        self.db.commit()

def determine_split(split_probs):
    if not split_probs['train'] + split_probs['val'] + split_probs['test'] == 100:
        raise AssertionError("Splits don't sum to 100%")
    rand_num = random.randint(1, 100)
    if rand_num < split_probs['train']:
        split = 'train'
    elif rand_num <= (split_probs['train'] + split_probs['val']):
        split = 'val'
    else:
        split = 'test'
    return(split)

def create_dir_if_not_exist(directory):
    """Creates a directory if the path does not yet exist.

    Args:
        directory (string): The directory to create.
    """          
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_random_filename(length=10):
    # https://www.pythoncentral.io/python-snippets-how-to-generate-random-string/
    allchar = string.ascii_letters + string.digits
    return("".join(random.choice(allchar) for x in range(length)))

def create_dataloader_folders(root_dir, data_folder_name, classes):
    """For a given root dir, creates a folder structure which matches the PyTorch dataloader format"""
    data_root = os.path.join(root_dir, data_folder_name)
    create_dir_if_not_exist(data_root)
    splits = ['train', 'val', 'test']
    for split in splits:
        split_folder = os.path.join(data_root, split)
        create_dir_if_not_exist(split_folder)
        for c in classes:
            class_folder = os.path.join(split_folder, c)
            create_dir_if_not_exist(class_folder)

def randomly_sample_from_folder(folder_in, folder_out, retain_every_n, seed=1):
    """For a given folder, shuffles the folder and retains every nth image"""
    random.seed(seed)
    files_in_root = next(os.walk(folder_in))[2]
    random.shuffle(files_in_root)
    create_dir_if_not_exist(folder_out)
    n_imgs = 0
    for i, image in enumerate(files_in_root):
        if (n_imgs+1) % retain_every_n == 0:
            if is_image(image):
                in_img_path = os.path.join(folder_in, image)
                out_img_path = os.path.join(folder_out, image)
                shutil.copy(in_img_path, out_img_path)
                n_imgs += 1

def sample_n_images_from_dir(dir_in, target_dir, retain_n_total, seed=1):
    create_dir_if_not_exist(target_dir)
    random.seed(seed)
    images = []
    for root, _, files in os.walk(dir_in):
        images += [os.path.join(root, file) for file in files if is_image(file)]
    random.shuffle(images)
    if retain_n_total > len(images):
        retain_n_total = len(images)
        print(f'Not enough images in the set, retaining all ({retain_n_total})')
    selected_imgs = random.sample(images,retain_n_total)
    return selected_imgs

def get_coord_splits(imgs_with_coords, split_probs, dim_range, dim_index):
    """Distributes images into geographic regions using an ascending coordinate axis
    """
    split_entries = {}
    upper_limit_multiplier = 0 # Running increment for indexing
    sorted_imgs = sorted(imgs_with_coords, key=operator.itemgetter(5))
    i = 0
    for split in split_probs:
        split_entries[split] = []
        upper_limit_multiplier += split_probs[split]
        upper_limit = int(upper_limit_multiplier * dim_range[1])
        while float(imgs_with_coords[i][dim_index]) <= upper_limit:
            split_entries[split].append(sorted_imgs[i])
            i += 1
        i = upper_limit+1 # set lower limit
    return split_entries

def get_geograph_coords(geograph_img_filepaths, geograph_metadata_file, class_index):
    """Retrieves all geograph entries that match the filepath in the metadata file"""
    with open(geograph_metadata_file, 'r', encoding='utf8', errors='ignore') as meta_file:
        metadata = csv.reader(meta_file, delimiter=',')
        return [entry for entry in metadata if entry[class_index] in geograph_img_filepaths]