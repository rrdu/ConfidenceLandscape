#utils/landscape.py
'''Makes the confidence landscape, runs perturbations'''

import torch
import numpy as np
from .perturbations import PERTURB_FUNCS
#############################################################
@torch.no_grad()
def build_confidence_landscape(
    model,
    base_img_pil,
    class_idx:int,
    x_axis:str,
    y_axis:str,
    x_vals,
    y_vals,
    preprocess,
    device='cpu',
    batch_size=32
):
    print(f'[LANDSCAPE] Starting to build...')
    model = model.to(device)
    model.eval()

    imgs = []
    confs = []

    for x_value in x_vals:
        for y_value in y_vals:
            #Normalize to lowercase
            x_key = x_axis.lower()
            y_key = y_axis.lower()

            #Apply perturbation to x then y axis
            img_xy = PERTURB_FUNCS[x_key](base_img_pil, float(x_value))
            img_xy = PERTURB_FUNCS[y_key](img_xy, float(y_value))

            t = preprocess(img_xy)
            imgs.append(t)

            if len(imgs) == batch_size:
                batch = torch.stack(imgs).to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)[:, class_idx]
                confs.extend(probs.cpu().numpy().tolist())
                imgs = []
        
    if imgs: 
        batch = torch.stack(imgs).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, class_idx]
        confs.extend(probs.cpu().numpy().tolist())

    #Reshape to (len(x), len(y))
    Z = np.array(confs).reshape(len(x_vals), len(y_vals))

    print(f'[LANDSCAPE] Done building landscape')
    
    return Z