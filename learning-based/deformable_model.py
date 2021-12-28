"""
File: deformable_model.py  
-------------------
This file contains the PyTorch source code for the deformable 
learning-based registration model described in section 2.3.1. 
""" 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

INPUT_DIM = 7864320

class reg_net(nn.Module):
    def __init__(self):
        super(reg_net, self).__init__()
        
        # localization network 
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
        self.dec5 = nn.Linear(in_features=256, out_features=6) # 2x3 affine matrix 

    def forward(self, fixed_img, moving_img):
        # localization network 
        # encoder
        concat_np = np.concatenate((fixed_img, moving_img), axis=None)
        x = torch.tensor(concat_np, dtype=torch.float32)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        # decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        
        # affine matrix 
        theta = F.relu(self.dec5(x)) 
        theta = theta.view(-1, 2, 3) 

        # displacement field 
        grid_shape = ((fixed_img.unsqueeze(0)).unsqueeze(0)).size()
        displacement_grid = F.affine_grid(theta, grid_shape)

        # warp 
        moving_img = torch.tensor(moving_img, dtype=torch.float32)
        moving_img = moving_img.unsqueeze(0)
        moving_img = moving_img.unsqueeze(0)
        deformed_image = F.grid_sample(moving_img, displacement_grid)
        deformed_image_reshaped = torch.squeeze(deformed_image)
        
        return deformed_image_reshaped

