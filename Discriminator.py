from pyexpat import model
import torch
import functools
from torch._C import _set_qengine
from torch.functional import norm
import torch
import torch.nn as nn
import numpy as np
import math
import torchvision.models as models


class PatchGAN(nn.Module):
    def __init__(self, input_size, ndf=64, isPatch=False):
        super(PatchGAN, self).__init__()
        self.k_size = 4
        self.padding = 'same'
        self.net = [
            nn.Conv2d(3, 64, kernel_size=self.k_size, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=self.k_size, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=self.k_size, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=self.k_size, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=self.k_size, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.net = nn.Sequential(*self.net)
    
    def forward(self, input):
        return self.net(input)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding, stride, Instacne_layer=True):
        super(ConvBlock, self).__init__()
        self.network = [
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding),
        ]
        if Instacne_layer :
            self.network.append(nn.InstanceNorm2d(out_c))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self, input_size=3, ndf=64):
        super(Discriminator, self).__init__()
        self.k, self.s, self.p = 4, 2, 1
        self.conv1 = ConvBlock(input_size, ndf, self.k, self.p, self.s, False)
        self.conv2 = ConvBlock(ndf, ndf*2, self.k, self.p, self.s)
        self.conv3 = ConvBlock(ndf*2, ndf*4, self.k, self.p, self.s)
        self.conv4 = ConvBlock(ndf*4, ndf*4, self.k, self.p, self.s)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf*4*4, ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Linear(ndf*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #input = nn.functional.interpolate(input, (256,256), mode='bilinear')
        #print(input.shape).
        """
        x: (batch_size, 3, H, W)
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.mlp(x4)
        return out

#https://arxiv.org/abs/1807.03247
class AddCoords(nn.Module):
    def __init__(self, with_r):
        super(AddCoords, self).__init__()
        self.with_r = with_r
    
    def forward(self, input_tensor):
        """
        input_tensor: (batch, channel, x_dim, y_dim)
        """
        batch_size, _, self.x_dim, self.y_dim = input_tensor.shape
        print(batch_size.shape)
        xx_ones = torch.ones([batch_size, self.x_dim], dtype=torch.float32).unsqueeze(-1) #[n, x_dim, 1]
        xx_range = torch.arange(self.y_dim).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1) #[n, 1, y_dim]
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1) #[n, 1, x_dim, y_dim]
        print(xx_channel.shape)

        yy_ones = torch.ones([batch_size, self.y_dim], dtype=torch.float32).unsqueeze(1) #[n, 1, y_dim]
        yy_range = torch.arange(self.x_dim).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) #[n, x_dim, 1]
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1) #[n, 1, x_dim, y_dim]
        print(yy_channel.shape)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = torch.cat([input_tensor, xx_channel, yy_channel], axis=1) #[n, c+2, x_dim, y_dim]
        print(ret.shape)
        if self.with_r:
            r = torch.sqrt(torch.square(xx_channel) + torch.square(yy_channel))
            ret = torch.cat([ret, r], axis=1)
        return ret


class CoordConv(nn.Module):
    def __init__(self, in_c, out_c, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_c = in_c + 3 if with_r else in_c + 2
        self.conv = nn.Conv2d(in_c, out_c, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

class ResidualCoordConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, downsample=False):
        super(ResidualCoordConv, self).__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(out_c, out_c, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #torch.nn.init.kaiming_normal_(self.network.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        self.proj = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else None
        self.downsample = downsample

    def forward(self, x):
        y = self.network(x)
        if self.downsample:
            y = nn.funtional.avg_pool2d(y, 2)
        if self.downsample:
            x = nn.functional.avg_pool2d(x, 2)
        x = x if self.proj is None else self.proj(x)
        y = (y + x) / math.sqrt(2)
        return y

class ProgressiveDiscriminator(nn.Module):
    def __init__(self):
        super(ProgressiveDiscriminator, self).__init__()
        self.layers = nn.ModuleList([
            ResidualCoordConv(16, 32, downsample=True), 
            ResidualCoordConv(32, 64, downsample=True),
            ResidualCoordConv(64, 128, downsample=True),
            ResidualCoordConv(128, 256, downsample=True),
            ResidualCoordConv(256, 400, downsample=True),
            ResidualCoordConv(400, 400, downsample=True),
            ResidualCoordConv(400, 400, downsample=True),
            ResidualCoordConv(400, 400, downsample=True),
        ])
        
        self.fromRGB = nn.ModuleList([
            ConvBlock(3, 16, 1, 0, 1, False),
            ConvBlock(3, 32, 1, 0, 1, False),
            ConvBlock(3, 64, 1, 0, 1, False),
            ConvBlock(3, 128, 1, 0, 1, False),
            ConvBlock(3, 256, 1, 0, 1, False),
            ConvBlock(3, 400, 1, 0, 1, False),
            ConvBlock(3, 400, 1, 0, 1, False),
            ConvBlock(3, 400, 1, 0, 1, False),
            ConvBlock(3, 400, 1, 0, 1, False),
        ])
        
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}
    
    def forward(self, input, alpha, instance_noise=0, **kwargs):
        start_idx = self.img_size_to_layer[input.shape[-1]]

        x = self.fromRGB[start_idx](input)
        for i, layer in enumerate(self.layers[start_idx:]):
            if i == 1:
                x = alpha * x + (1-alpha) * self.fromRGB[start_idx+1](nn.functional.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x

class VGGDiscriminator(nn.Module):
    def __init__(self):
        super(VGGDiscriminator, self).__init__()
        self.network = models.vgg19(pretrained=True)
        self.final_layer = nn.Sequential(
            nn.Linear(1000, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.network(x)
        x = self.final_layer(x)
        return x
