from numpy.core.fromnumeric import transpose
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as tranforms
import cv2
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import random
import os

blur_path = './nerf_pytorch/data/nerf_llff_data/horns/images_8_random'
blur_imgs = [os.path.join(blur_path, f) for f in sorted(os.listdir(blur_path))]

gaussain_t = transforms.Compose([
    transforms.ToPILImage(),
    transforms.GaussianBlur(kernel_size=51, sigma=10),
])

inpainting_t = transforms.Compose([
    transforms.ToPILImage(),
])

toPIL_t = transforms.Compose([
    transforms.ToPILImage(),
])

def preprocessing(imgs, mode):
    masks = []
    for i in range(imgs.shape[0]):
        
        image = torch.Tensor(imgs[i])
        image = image.permute((2,0,1))
        if mode == 'Random':
            p_type = random.choice(["Blur", "Noise", "Inpainting"])
        else:
            p_type = mode


        

        if p_type == 'Blur':
            image = Image.open(blur_imgs[i])
            
        elif p_type == 'Noise':
            image = torch.clip(image + torch.randn(image.shape) * 0.2 + 0.0, 0,1)
            image = toPIL_t(image)
        elif p_type == 'Inpainting':
            mask = torch.ones_like(image)
            mask_h, mask_w = int(image.shape[1] / 2), int(image.shape[2] / 2)
            x = int(image.shape[1] / 4)
            y = int(image.shape[2] / 4)
            mask[:, x:x+mask_h, y:mask_w+y] = 0
            image = image * mask
            image = toPIL_t(image)    
            masks.append(torch.Tensor(mask).unsqueeze(0))
        else:
            print("Not defined type")
            image = toPIL_t(image)


        reverse_t = transforms.ToTensor()

        print('save png')
        image.save('./preprocess/preprcessed_{}.png'.format(i))
        image = reverse_t(image).permute((1, 2, 0)).numpy()
        print(image.shape)
        imgs[i] = image
    masks = torch.cat(masks, 0).permute(0, 2, 3, 1)
    return imgs, masks
