import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def create_samples(path_in, path_out):
#    init_names_in = os.listdir(path_in) # List all sub-directories in in_path
#    init_names_out = os.listdir(path_out)
#    if nat_sort:
#        sub_dir_names_in = natsorted(init_names_in) # sort directory names in natural order
#                                              # (Only for directories with numbers for names)
#        sub_dir_names_out = natsorted(init_names_out)
#    else:
#        sub_dir_names_in = init_names_in
#        sub_dir_names_out = init_names_out
    data_samples = []
#    for sub_dir in sub_dir_names: # Loop over all sub-directories
#        per_dir = os.listdir(root+'/'+sub_dir) # Get a list of names from sub-dir # i

    RGB_list = []
    pl_list = []
    image_num = 0
    for name in path_in:
        image_num = image_num + 1
#        split_name = name.split('_')
        
        RGB_list = (path_in + '/' + 'image' + '/' + str(image_num))
        pl_list = (path_out + '/' + 'image' + '/' + str(image_num))
        sample = (RGB_list,pl_list)
        data_samples.append(sample)
    

    return data_samples
    










class Dataset_Pro9(Dataset):
    def __init__(self, dep_dir, RGB_dir, pl_dir, transform=None):
        self.dep_dir = dep_dir
        self.RGB_dir = RGB_dir
        self.pl_dir = pl_dir
        self.dep_images = sorted(os.listdir(dep_dir))
        self.RGB_images = sorted(os.listdir(RGB_dir))
        self.pl_images = sorted(os.listdir(pl_dir))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),            # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        self.transform_pl = transforms.Compose([
            transforms.Resize((64, 64)),  # 根据需求调整尺寸
            transforms.ToTensor(),           # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __len__(self):
        return len(self.dep_images)

    def __getitem__(self, idx):
        dep_path = os.path.join(self.dep_dir, self.dep_images[idx])
        RGB_path = os.path.join(self.RGB_dir, self.RGB_images[idx])
        pl_path = os.path.join(self.pl_dir, self.pl_images[idx])
        
        dep_image = Image.open(dep_path).convert('RGB')  
        RGB_image = Image.open(RGB_path).convert('RGB')
        pl_image = Image.open(pl_path).convert('L')    
        
        dep_image = self.transform(dep_image)
        #print('dep_image:',dep_image.shape)
        RGB_image = self.transform(RGB_image)
        #print('rgb_image:',RGB_image.shape)
        pl_image = self.transform_pl(pl_image)
        #print('pl_image:',pl_image.shape)
        
        return dep_image, RGB_image, pl_image
    

