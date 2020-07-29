import torch
from torch.utils.data import DataLoader, Dataset

#mean= tensor([0.5271, 0.5788, 0.6095])
#std= tensor([0.1707, 0.1650, 0.1804])
#(tensor([0.5271, 0.5788, 0.6095]), tensor([0.1707, 0.1650, 0.1804]))

def mean_std(dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=True, num_workers=1)
  mean = torch.zeros(3)
  std = torch.zeros(3)
  print('Computing mean and std from dataloader')
  for inputs in dataloader:
	  for i in range(3):
		  mean[i] = mean[i] + inputs[:,i,:, :].mean()
		  std[i]  = std[i] + inputs[:,i,:, :].std()
  mean.div_(len(dataset))
  std.div_(len(dataset))
  print('mean=',mean)
  print('std=',std)
  return mean, std
