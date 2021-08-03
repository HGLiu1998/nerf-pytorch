from numpy.core.fromnumeric import transpose
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as tranforms
import cv2
import numpy as np
from torchvision.transforms import transforms

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def preprocessing(imgs):
    for i in range(imgs.shape[0]):
        image = torch.Tensor(imgs[i])
        image = image.permute((2,0,1))
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=51, sigma=2),
            #transforms.GaussianBlur((51,51))
        ])
        reverse_t = transforms.ToTensor()   
        image = t(image)

        print('save png')
        image.save('51-2.png')
        image = reverse_t(image).permute((1, 2, 0)).numpy()
        imgs[i] = image
    return imgs

