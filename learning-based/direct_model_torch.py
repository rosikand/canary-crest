"""
File: direct_model_torch.py  
-------------------
This file contains the PyTorch source code for the direct, generative 
learning-based registration model. This model was not included in the 
report. In addition to the model, the source code for the dataset class
as well as the training and loss function (Dice) is included. 
""" 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, TensorDataset
from matplotlib import pyplot as plt 

INPUT_DIM = 7864320
OUTPUT_DIM = 6291456

class reg_net(nn.Module):
    def __init__(self):
        super(reg_net, self).__init__()

        # Autoencoder src adapted from: 
        # https://debuggercafe.com/implementing-deep-autoencoder-in-pytorch/ 
        # encoder
        self.enc1 = nn.Linear(in_features=INPUT_DIM, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=OUTPUT_DIM) # notice out_features is same as 1st layer in_features

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x 


class TrainDataset(Dataset):
    def __init__(self, fixed_images, moving_images): # samples and labels are lists
        self.X_fixed = torch.tensor(fixed_images)
        self.X_moving = torch.tensor(moving_images)
    
    def __getitem__(self, index):
        return (self.X_fixed[index], self.X_moving[index])
    
    def __len__(self):
        return len(self.X_fixed)


def calculate_dice(array_one, array_two):
    """
    Calculates the Dice Coefficient of two Numpy 
    arrays as inputs. If wanting to input images,
    convert them into Numpy arrays through a package
    such as MedPy, Pillow, or OpenCV. 
    """
    
    # calculate the numerator 
    numerator_intersection = np.intersect1d(array_one, array_two)
    numerator_cardinality = np.size(numerator_intersection)
    numerator_final = 2 * numerator_cardinality

    # calculate the denominator
    x_cardinality = np.size(array_one)
    y_cardinality = np.size(array_two)
    denominator_final = x_cardinality + y_cardinality

    # return the division
    return numerator_final/denominator_final


def train(epoch_num):
    # Training loop 

    loss_values = [] # for plot
    outputs_list = []
    for epoch in range(epoch_num):  # loop over the dataset multiple times

        running_loss = 0.0 
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            fixed_sample = data[0][0]
            moving_sample = data[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            concat_np = np.concatenate((fixed_sample, moving_sample), axis=None)
            outputs = net(torch.tensor(concat_np, dtype=torch.float32))
            
            # show outputs
            plt.imshow(outputs.detach().numpy().reshape(28,28), interpolation='nearest')
            plt.show() 
            
            loss = calculate_dice(outputs.detach().numpy(), fixed_sample.detach().numpy())
            outputs_list.append(outputs)
            loss = torch.tensor(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            loss_values.append(loss)

    print('Finished Training')

