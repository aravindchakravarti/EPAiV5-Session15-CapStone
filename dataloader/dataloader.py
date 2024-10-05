# dataloader/dataloader.py

import os
import sys
import requests
from collections import namedtuple
from contextlib import contextmanager
from .preprocessors import default_preprocess
from .utils import download_file, timer
from .utils import read_idx, unpickle
import numpy as np
import matplotlib.pyplot as plt

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

        # print(f"Data after pre-processing = {(self.data[0])}")
    
    @timer
    def download_dataset(self):
        # Implement dataset download logic
        print(f"Downloading {self.dataset_name} dataset...")
        if self.dataset_name == 'MNIST':
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

        elif self.dataset_name == 'CIFAR-10':
            os.makedirs(f'datasets/{self.dataset_name}')
            urls = {
                'data' : 'https://drive.google.com/uc?export=download&id=1Nh71Y_31pP2qu4KuNMqBavBg7h-WDMNm'
            }
            dest_path = f'datasets/{self.dataset_name}/file.tar'
            download_file(urls['data'], dest_path)

    
    def read_data(self):
        # Implement data reading logic
        if self.dataset_name == 'MNIST': 
            dataset_to_read = {
                'train_images' : f'datasets/{self.dataset_name}/train_images/file',
                'train_labels' : f'datasets/{self.dataset_name}/train_labels/file',
                'test_images' : f'datasets/{self.dataset_name}/test_images/file',
                'test_labels' : f'datasets/{self.dataset_name}/test_labels/file'
            }

            for type, path in dataset_to_read.items():
                if os.path.exists(path):
                    # TO-DO : We should do it for every image
                    if type == 'train_images':
                        train_data = read_idx(path)
                        # print(f'type = {train_data.shape}')
                    elif type == 'train_labels':
                        train_labels = read_idx(path)
                        # print(f'type = {train_labels.shape}')
                else:
                    raise ValueError(f"In {self.dataset_name} {type} doesn't exists")
            
            # print(f'Data before preprocessing = {train_data[0]}')
            # print("\n*********************************************\n")
            return train_data
        
        elif self.dataset_name == 'CIFAR-10':
            cifar_path = f'datasets/{self.dataset_name}/cifar-10-batches-py'
            all_files = os.listdir(cifar_path)
            train_img_files = [f for f in all_files if f.startswith('data_batch_')]
            train_data = []
 
            for img_batch in train_img_files:
                data_dict = unpickle(os.path.join(cifar_path,img_batch))
                img_data = data_dict[b'data']
                img_label = data_dict[b'labels']

                img_data = img_data.reshape(-1, 3, 32, 32)  # Reshape to (batch_size, 3, 32, 32)
                img_data = np.transpose(img_data, (0, 2, 3, 1))  # Transpose to (batch_size, 32, 32, 3)

                ''' 
                To confirm whther images are read properly or not, 
                uncomment below lines, which will display first image in each batch
                '''
                # plt.imshow(img_data[0])
                # plt.show()
                # print(f'image data = {img_data.shape}, label = {len(img_label)}')
                
                train_data.append(img_data.tolist())
            
            print(f'Length of CIFAR-10 data = {len(train_data[0])}')
            
            return train_data
    
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
