#utils/cache.py

'''Creates a cache to avoid making a new grid every time'''

import os
import json 
import hashlib
import numpy as np

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
#############################################################
def _config_to_key(config:dict) -> str:
    '''Make unique cache key from config dictionary'''
    s = json.dumps(config, sort_keys=True)

    return hashlib.md5(s.encode()).hexdigest()
#############################################################
def load_from_cache(config:dict):
    '''Load a cached numpy array that matches a configuration'''
    key = _config_to_key(config)
    path = os.path.join(CACHE_DIR, key + '.npy')
    if os.path.exists(path):
        return np.load(path), key

    return None, key

#############################################################
def save_to_cache(key:str, arr):
    '''Save a numpy array to the disk using a given cache key'''
    path = os.path.join(CACHE_DIR, key + '.npy')
    np.save(path, arr)