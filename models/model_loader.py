#models/model_loader.py
'''Loads the 3 models of interest'''

from torchvision import models

#############################################################
def load_resnet18():
    '''Load in pretrained ResNet18 model'''
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    model.eval()

    return model
#############################################################
def load_mobilenet_v3_large():
    '''Load in pretrained MobileNet V3 Large model'''
    model = models.mobilenet_v3_large(
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
    )
    model.eval()

    return model
#############################################################
def load_efficientnet_b0():
    '''Load in pretrained MobileNet V3 Large model'''
    model = models.mobilenet_v3_large(
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
    )
    model.eval()

    return model
#############################################################