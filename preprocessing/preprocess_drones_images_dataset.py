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
#This particular class is used for preprocessing.
#Take all types of images supported by PIL and resize to 224*224 and save to new parent folder.
  
#Considered 4 folders (n). Take all the files from those folders into a list, a global list in memory.     
#Flying birds, Large Quadcopters, Small QuadCopters, Winged drones
#Given an index, list[index] will be returned as a file.
root_dir = "/content/Session2Dataset"
class PreprocessImageFolderDataset(Dataset):

  def __init__(self, root_dir, transform=ToTensor()):
    'Initialization'
    self.m_root = root_dir
    self.m_transform = transform
    logging.basicConfig(filename = "/content/dataloader_timeit.log",datefmt='%H:%M:%S',level=logging.DEBUG)
    #self.g_dict_fileindex = dict()
    self.g_array_fileindex1 = []
    self.walkTheDirectory(self.m_root) 
    self.m_length = len(self.g_array_fileindex1)
  
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
    subdirs = [x[0] for x in os.walk(root_dir)]
    dir = ''
    try:
      dir = subdirs[0] + "-copy"
      os.mkdir(dir)
    except:
      print('present')

    count = -1
    dir1 =  subdirs[0] + "-copy"
    print('dir1=',dir1) 
    for d in subdirs:
      count = count + 1
      if count == 0:
        continue
      basedir = os.path.basename(subdirs[count])
      dir = os.path.join(dir1,basedir)
      print(dir)
      try:
        os.mkdir(dir)
      except:
        print('present')
    
    for path, subdirs, files in os.walk(self.m_root): # path should give the complete path till the last directory.
        for filename in files:
            #print(os.path.join(path, filename))
            full_path = os.path.join(path, filename)
            #self.g_dict_fileindex[index] = full_path
            self.g_array_fileindex1.append(full_path)
            
 
  def __len__(self):
    'Denotes the total number of samples'
    return self.m_length;

  def printItem(self,idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
    file = self.g_array_fileindex1[idx]
    print('printFile-',file)  
  
  def removeItem(self,idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
    #self.g_array_fileindex.remove(file)

  def __getitem__(self, idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
        
    start_time = timeit.default_timer()

    # Pick the file from the index
    size = 224, 224
    file = self.g_array_fileindex1[idx]
    
    """
    try:
      im = Image.open(file)
      im.resize(size)
      im.save(save)
    except IOError:
      print("cannot create thumbnail for", file)
    """
    new_dir = "/content/Session2Dataset-copy"
    file1 = file
    try:
      image = Image.open(file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
      #temporarily we will resize to 224*224
      # We need to look into meaningful way of resizing (...pyramid pooling??)
      image = image.resize((224,224), Image.ANTIALIAS)
      filename, file_extension = os.path.splitext(file)
      basedir = os.path.basename(os.path.dirname(filename))
      #append subdir to new dir
      dir1 = os.path.join(new_dir,basedir) # This is new directory path.
    
      #Create jpg file
      basefile=os.path.basename(file)#This comes with extension
      #split with extension
      file1, ext = os.path.splitext(basefile)
      file1 = file1 + ".jpg"
      file1 = os.path.join(dir1,file1)#complete new file path
      print('new filename=',file1)
      image.save(file1)
      
      if self.m_transform:
        image = self.m_transform(image)
      
      self.g_array_fileindex1[idx] = file1
      
    except:
      print("exception while opening=",file)
      
    load_time = timeit.default_timer() - start_time
    
    desc = f' file={file1} size={image.size} LOAD_TIME={load_time:0.3f}'
    logging.info(desc)
      
    return image
    
import torch
import torchvision.transforms

#import WingsFolderDataset as wingsDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root_dir = "/content/Session2Dataset"

tr = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(),
    transforms.ToTensor()
    #normalize,
    #transforms.RandomErasing()
])

data_set = PreprocessImageFolderDataset("/content/Session2Dataset", transform = tr)

means = []
stds = []
print('length of dataset=',len(data_set))
for i in range(len(data_set)):
    if i == 1000 or i or 5000 or i == 10000 or i == 15000 or i == 20000:
      print('i=',i)
    
    try:
      img  = data_set[i]
      means.append(torch.mean(img))
      stds.append(torch.std(img))
    except:
      data_set.printItem(i)
      
    #print(torch.mean(img))
    
mean = torch.mean(torch.tensor(means))
std = torch.mean(torch.tensor(stds))

print('mean=',mean)
print('std=',std)
#mean= tensor(0.5724)
#std= tensor(0.1992)


