import math
from unittest import result
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.l1 = nn.L1Loss()
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False
    
    def forward(self, output, target):
        loss = 0
        for i in range(3):
            func = getattr(self, 'enc_{}'.format(i+1))
            loss += self.l1(func(output), func(target))
            output, target = func(output), func(target)
        return loss

def get_mask_coords(imgs, masks, H, W, cur_shape):
    maskCoords = []
    for i in range(imgs.shape[0]):
        mask = nn.functional.interpolate(masks[i], size=[cur_shape, cur_shape])
        mask = torch.Tensor(mask)
        mask_coords = []
        for h in range(H):
            for w in range(W):
                if not torch.equal(mask[h, w, :], torch.Tensor([1,1,1])):
                    mask_coords.append(torch.FloatTensor([h, w]).unsqueeze(0))
        mask_coords = torch.cat(mask_coords, 0)
        maskCoords.append(mask_coords)

    return maskCoords

def fixed_coords(cur_shape, k):
    x = np.array([i for i in range(k//2, cur_shape, k)])
    y = np.array([i for i in range(k//2, cur_shape, k)])
    coord_pairs = []
    for i in range(len(x)):
        for j in range(len(y)):
            coord_pairs.append(np.array([x[i], y[j]]))
    return coord_pairs
