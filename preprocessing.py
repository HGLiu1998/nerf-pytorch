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

toTensor = tranforms.Compose([
    tranforms.ToTensor(),
])

scale = transforms.Compose([
    transforms.Resize((512, 512))
])

def preprocessing(imgs, datadir, using_mask=True, scale=False):
    mask_path = datadir + '/images_8_mask'
    mask_imgs = [os.path.join(mask_path, f) for f in sorted(os.listdir(mask_path))]
    masks = []
    i_train = [i for i in range(imgs.shape[0])]
    #for i in i_train:
    #    if(i % 8 == 0):
    #        i_train.remove(i)
    #i_train.remove(1)
    if scale:
        original_imgs = np.zeros((imgs.shape[0], 512, 512, 3))
        for i in range(imgs.shape[0]):
            original_imgs[i] = cv2.resize(imgs[i], dsize=(512,512), interpolation=cv2.INTER_NEAREST)
        imgs = original_imgs.copy()
    else:
        original_imgs = imgs.copy()
    
    for i in range(imgs.shape[0]):

        image = torch.Tensor(imgs[i])
        image = image.permute((2,0,1))
        
        mask = torch.ones_like(image)
        
        if i in i_train:
            if using_mask == True:
                
                m_img = Image.open(mask_imgs[i])
                m_img = toTensor(m_img).cuda()
                if scale:
                    m_img = scale(m_img)
                #mask = m_img[:3, :, :]
                for n in range(m_img.shape[1]):
                    for m in range(m_img.shape[2]):
                        if torch.equal(m_img[:3, n, m], torch.Tensor([0,0,0])):
                            mask[:, n, m] = torch.Tensor([0,0,0])
                        else:
                            mask[:, n, m] = torch.Tensor([1,1,1])
                #print(torch.sum(mask[0,:,:]))
                
                image = image * mask
                image = toPIL_t(image)
            else:
                mask = torch.ones_like(image)
                mask_h, mask_w = int(image.shape[1] / 4), int(image.shape[2] / 4)
                
                x = random.randint(0, int(image.shape[1] - image.shape[1]/4))
                y = random.randint(0, int(image.shape[2] - image.shape[2]/4))
                mask[:, x:x+mask_h, y:mask_w+y] = 0.0
                image = image * mask
                image = toPIL_t(image)      
        else:
            print("Not in training set")
            image = toPIL_t(image)

        
        reverse_t = transforms.ToTensor()
        print('save png')
        image.save('./preprocess/preprcessed_{}.png'.format(i))
        image = reverse_t(image).permute((1, 2, 0)).numpy()
        print(image.shape)
        imgs[i] = image
        masks.append(torch.Tensor(mask).unsqueeze(0))

    masks = torch.cat(masks, 0).permute(0, 2, 3, 1)
    return imgs, masks, original_imgs
