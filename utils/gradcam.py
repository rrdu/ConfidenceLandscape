#utils/gradcam.py

'''Implements GradCAM for the chosen image'''

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize

#############################################################
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.fmap = None
        self.grad = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

        #Save forward features
        target_layer.register_forward_hook(self._forward_hook)

        #Save backward grads
        target_layer.register_backward_hook(self._backward_hook)
    ########################################################
    def _forward_hook(self, m, input, output):
        self.fmap = output
    #########################################################
    def _backward_hook(self, m, g_input, g_output):
        self.grad = g_output[0]
    #########################################################
    def generate(self, x, class_idx:int):
        self.model.zero_grad()
        out = self.model(x)
        score = out[:, class_idx].sum()
        score.backward()

        #Weights = global average
        weights = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        #Normalize
        cam = cam / (cam.max() + 1e-6)

        #Upsample to match input
        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        cam = cam[0,0].detach().cpu().numpy()
        
        return cam