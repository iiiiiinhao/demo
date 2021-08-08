#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Warper3d(nn.Module):
    def __init__(self, img_size):
        super(Warper3d, self).__init__()
        """
        warp an image, according to the optical flow
        image: [B, 1, H, W, D] image for sampling
        flow: [B, 3, H, W, D] flow predicted from source image pair
        """
        self.img_size = img_size

        H, W, D =img_size
        print(H,W,D)
        
        #create a grid
        xx = torch.arange(0, H).view(-1, 1, 1).repeat(1, W, D)
        yy = torch.arange(0, W).view(1, -1, 1).repeat(H, 1, D)
        zz = torch.arange(0, D).view(1, 1, -1).repeat(H, W, 1)

        self.grid = np.zeros([1, H, W, D, 3])

        #Normalized，same with the way of the training image set
        xx1 = 2.0 * xx / H - 1  # max(W-1,1)
        yy1 = 2.0 * yy / W - 1  # max(H-1,1)
        zz1 = 2.0 * zz / D - 1  # max(H-1,1)

        self.grid[0, :, :, :, 0] = zz1
        self.grid[0, :, :, :, 1] = yy1
        self.grid[0, :, :, :, 2] = xx1

            
    def forward(self, img, flow):


        grid = self.grid
        grid = torch.tensor(grid)

        if img.is_cuda:
            grid = grid.cuda()

        #Add grid and deformation field data
        flow = flow.permute(0, 2, 3, 4, 1)  # [bs, D, H, W, 3]
        vgrid = grid + flow
        img = img.to(torch.float64)

        #Re-sampling
        output = F.grid_sample(img, vgrid, padding_mode='border', align_corners=True)  # , mode='nearest'

        return output
