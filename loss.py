import math
from unittest import result
import torch
import torch.nn as nn
from torchvision import models

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