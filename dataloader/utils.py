# dataloader/utils.py
import requests
import time
import gzip
import shutil
import os
import gdown
import struct
import numpy as np

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

    if dest_path.endswith('.gz'):
        with gzip.open(dest_path, 'rb') as f_in:
            with open(dest_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Uncomment if you want to delete the zip file
        # os.remove(dest_path)

        print(f'Data extracted successfully for file {dest_path[:-3]}')

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
