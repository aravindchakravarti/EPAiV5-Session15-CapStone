# dataloader/dataloader.py

import os
import sys
import requests
from collections import namedtuple
from contextlib import contextmanager
from .preprocessors import default_preprocess
from .utils import download_file, timer
from .utils import read_idx

DataSample = namedtuple('DataSample', ['features', 'label'])

class DataLoader:
    def __init__(self, dataset_name='MNIST', batch_size=32, shuffle=True, **kwargs):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.data = []
        self.index = 0
        self.load_data()
    
    @timer
    def load_data(self):
        if not os.path.exists(f'datasets/{self.dataset_name}'):
            self.download_dataset()
        # Implement data loading logic
        self.data = self.preprocess_data(self.read_data())
    
    @timer
    def download_dataset(self):
        # Implement dataset download logic
        print(f"Downloading {self.dataset_name} dataset...")
        urls = {
            'train_images': 'https://drive.google.com/uc?export=download&id=1ruFYL2hHgetFc6hLFE87aSS9GQQAgav9',
            'train_labels': 'https://drive.google.com/uc?export=download&id=1ILIdcDlpcs55lkQ1ycot58S4TQPu2_sx',
            'test_images': 'https://drive.google.com/uc?export=download&id=1AOW0gGEgQHU4EXrAG5o9m-UquA4R5aHP',
            'test_labels': 'https://drive.google.com/uc?export=download&id=12nE2NfMEz0SOVA0Pb_aIsGPvy6z9brw4'
        }

        for folder_name, url in urls.items():
            os.makedirs(f'datasets/{self.dataset_name}/{folder_name}')
            dest_path = f'datasets/{self.dataset_name}/{folder_name}/file.gz'
            download_file(url, dest_path)
    
    def read_data(self):
        # Implement data reading logic
        dataset_to_read = {
            'train_images' : f'datasets/{self.dataset_name}/train_images/file',
            'train_labels' : f'datasets/{self.dataset_name}/train_labels/file',
            'test_images' : f'datasets/{self.dataset_name}/test_images/file',
            'test_labels' : f'datasets/{self.dataset_name}/test_labels/file'
        }

        for type, path in dataset_to_read.items():
            if os.path.exists(path):
                data = read_idx(path)
                # print(f'type = {data.shape}')
            else:
                raise ValueError(f"In {self.dataset_name} {type} doesn't exists")
        
        return []
    
    def preprocess_data(self, data):
        # Implement data preprocessing logic
        preprocess_func = self.kwargs.get('preprocess_func', default_preprocess)
        return [preprocess_func(sample) for sample in data]
    
    def __iter__(self):
        self.index = 0
        if self.shuffle:
            import random
            random.shuffle(self.data)
        return self
    
    def __next__(self):
        if self.index < len(self.data):
            batch = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch
        else:
            raise StopIteration
