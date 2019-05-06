from os import path, makedirs

def create_dir_if_not_exist(directory):
    """Creates a directory if the path does not yet exist.

    Args:
        directory (string): The directory to create.
    """          
    if not path.exists(directory):
        makedirs(directory)
