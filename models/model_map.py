#models/model_map.py
'''Maps model name to appropriate loader'''

from .model_loader import(
    load_resnet18, 
    load_mobilenet_v3_large,
    load_efficientnet_b0
)
#############################################################
MODEL_MAP = {
    "ResNet18": load_resnet18,
    "Mobilenet_V3_Large": load_mobilenet_v3_large,
    "EfficientNet_B0": load_efficientnet_b0
}
#############################################################
def get_model(name:str):
    '''Load the correct model given the string name'''
    if name not in MODEL_MAP:
        raise ValueError(f"Invalid model: {name}")
    
    return MODEL_MAP[name]()