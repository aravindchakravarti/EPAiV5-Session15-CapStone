# dataloader/utils.py
import requests
import time
import gzip
import shutil
import os
import gdown
import struct
import numpy as np
import tarfile
from pathlib import Path
import pickle

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {time.time() - start_time:.2f}s to complete.")
        return result
    return wrapper

def download_file(url, dest_path):
    # Download the file 
    gdown.download(url, dest_path, quiet=False)
    print(f'Destination path = {dest_path}')
    if dest_path.endswith('.gz'):
        with gzip.open(dest_path, 'rb') as f_in:
            with open(dest_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Uncomment if you want to delete the zip file
        # os.remove(dest_path)

        print(f'Data extracted successfully for file {dest_path[:-3]}')
    
    elif dest_path.endswith('.tar'):
        print('Entered to Extract TAR file ***********')
        # Check if the file is a tar file
        if tarfile.is_tarfile(dest_path):
            with tarfile.open(dest_path) as tar:
                to_extact = Path(dest_path).parents[0]
                # Extract all contents to the specified directory
                tar.extractall(path=to_extact)
                print(f"Extracted to {to_extact}")
        else:
            print("The file is not a valid tar file.")

def read_idx(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        magic, size = struct.unpack(">II", f.read(8))  # ">II" reads two big-endian integers
        if magic == 2051:  # Magic number for image files
            rows, cols = struct.unpack(">II", f.read(8))
            # Read the data and reshape it into (size, rows, cols)
            data = np.fromfile(f, dtype=np.uint8).reshape(size, rows, cols)
        elif magic == 2049:  # Magic number for label files
            # Read the labels directly as a 1D array
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError("Invalid IDX file: unexpected magic number.")
    
    return data

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')  # CIFAR-10 is encoded as bytes
    return dict
