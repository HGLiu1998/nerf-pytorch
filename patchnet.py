import torch
import functools
from torch._C import _set_qengine
from torch.functional import norm
import torch
import torch.nn as nn
import numpy as np




class PatchNet(nn.Module):
    def __init__(self, input_size, ndf=64, isPatch=False):
        super(PatchNet, self).__init__()
        self.k_size = 4
        self.pad_size = 1
        self.midLayers = 3
        self.net = [
            nn.Conv2d(input_size, ndf, kernel_size=self.k_size, stride=2, padding=self.pad_size),
            nn.LeakyReLU(0.2, True)
        ]
        
        for i in range(self.midLayers):
            if i == self.midLayers - 1:
                self.net += [nn.Conv2d(ndf * (2**(i)), ndf * (2**(i+1)), kernel_size=self.k_size, stride=1, padding=self.pad_size)]
            else:
                self.net += [nn.Conv2d(ndf * (2**(i)), ndf * (2**(i+1)), kernel_size=self.k_size, stride=2, padding=self.pad_size)]
            self.net += [
                nn.InstanceNorm2d(ndf * (2 ** (i+1))),
                nn.LeakyReLU(0.2, True)
            ]
            if i == self.midLayers - 1:
                self.net += [nn.Conv2d(ndf * (2**(i+1)), 1, kernel_size=self.k_size, stride=1, padding=self.pad_size)]

        self.net = nn.Sequential(*self.net)

    
    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self, input_size, ndf=64):
        super(Discriminator, self).__init__()
        self.k, self.s, self.p = 4, 2, 1
        self.layers = 2
        self.net = [
            nn.Conv2d(input_size, ndf, self.k, self.s, self.p, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
  

        for i in range(self.layers):
            self.net += [
                nn.Conv2d(ndf * (2**(i)), ndf * (2**(i+1)), self.k, self.s, self.p, bias=False),
                nn.BatchNorm2d(ndf * (2 ** (i+1))),
                nn.LeakyReLU(0.2, True)
            ]
            if i == self.layers - 1:
                self.net += [
                    nn.Conv2d(ndf * (2**(i+1)), 1, self.k, 1, 0, bias=False),
                    nn.Flatten(),
                    nn.Sigmoid()
                ]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
