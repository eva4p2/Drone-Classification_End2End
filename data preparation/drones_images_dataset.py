import numpy as np
import cv2
import io
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
from PIL import Image
import sys
import torch.utils.data
import timeit
import logging

# This is the common class for both test and train data sets. Based on indexes, this class serves the transformed image tensor.
  
#Considered 4 folders (n). Take all the files from those folders into a list, a global list in memory.     
#Flying birds, Large Quadcopters, Small QuadCopters, Winged drones
#Given an index, list[index] will be returned as a file.
root_dir = "/content/Session2Dataset-copy"
class WingsFolderDataset(Dataset):

  def __init__(self, root_dir, transform=ToTensor()):
    'Initialization'
    self.m_root = root_dir
    self.m_transform = transform
    logging.basicConfig(filename = "/content/dataloader_timeit1.log",datefmt='%H:%M:%S',level=logging.DEBUG)
    #self.g_dict_fileindex = dict()
    self.g_array_fileindex = []
    self.walkTheDirectory(self.m_root) 
    self.m_length = len(self.g_array_fileindex)
  
  #root_dir structure
  #train
  #   /folder1
  #     file.jpg
  #     file40.jpg  
  #   /folder2
  #     filex.jpg
  #     filey.40.jpg  
  #   /folder3
  #     filea.jpg
  #     fileb.40.jpg  
  #   /folder4
  #     fileaa.jpg
  #     filebax.jpg  
  
  #test
  #   /anyfoldername1
  #     sfile.jpg
  #     mfile40.jpg  
  #   /foldername2
  #     efilex.jpg
  #     cfiley.40.jpg  
  #   /foldername3
  #     somefilea.jpg
  #     vfileb.40.jpg  
  #   /foldername4
  #     somefileaa.jpg
  #     afilebax.jpg  
  
  
    
  def walkTheDirectory(self,root_dir):
    #for root, dirs, files in os.walk('python/Lib/email'):
    # for file in files:
    #    with open(os.path.join(root, file), "r") as auto:    
    for path, subdirs, files in os.walk(self.m_root): # path should give the complete path till the last directory.
        for filename in files:
            #print(os.path.join(path, filename))
            full_path = os.path.join(path, filename)
            #self.g_dict_fileindex[index] = full_path
            self.g_array_fileindex.append(full_path)
            
 
  def __len__(self):
    'Denotes the total number of samples'
    return self.m_length;

  def printItem(self,idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
    file = self.g_array_fileindex[idx]
    print('printItem-',file)  

  def __getitem__(self, idx):

    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
        
    start_time = timeit.default_timer()

    # Pick the file from the index
    file = self.g_array_fileindex[idx]
    
    image = Image.open(file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
    size = image.size
    #image = image.resize((224,224)) # These are already resized images. Dont do preprocessing here..
    
    if self.m_transform:
      image = self.m_transform(image)
    
    load_time = timeit.default_timer() - start_time
    
    desc = f' file={file} size={image.size} LOAD_TIME={load_time:0.3f}'
    logging.info(desc)
      
    return image
    