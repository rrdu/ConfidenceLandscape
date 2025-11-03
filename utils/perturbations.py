#utils/perturbations.py
'''Stores the different perturbations that will be applied'''

import torch
import torchvision.transforms.functional as F

from PIL import Image, ImageEnhance

#############################################################
def apply_brightness(img:Image.Image, level:float) -> Image.Image:
    '''Brightens an image based on level (0,5x - 1.5x)'''
    return ImageEnhance.Brightness(img).enhance(level)

#############################################################
def apply_rotation(img:Image.Image, degree:float) -> Image.Image:
    '''Rotate an image based on degree'''
    return img.rotate(degree, resample=Image.Resampling.BILINEAR)

#############################################################
def apply_noise(img:Image.Image, sigma:float) -> Image.Image:
    '''Add noise to image'''
    t = F.to_tensor(img)
    t = t + sigma * torch.randn_like(t) #Gaussian noise
    t = t.clamp(0.0, 1.0) #Ensure valid intensity
    
    return F.to_pil_image(t)

#############################################################
PERTURB_FUNCS = {
    "brightness": apply_brightness,
    "rotation": apply_rotation,
    "noise": apply_noise,
}
