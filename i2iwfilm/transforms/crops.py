import numpy as np

from copy import deepcopy

from albumentations.core.transforms_interface import BasicTransform
from albumentations.augmentations.crops.transforms import CenterCrop

import matplotlib.pyplot as plt

class MyCenterCrop(BasicTransform):
    def __init__(self, height, width, keys, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

        self.keys = keys

    def __call__(self, *args, force_apply=False, **kwargs):
        # Apply the Center Crop on all keys in self.keys
        mod_kwargs = deepcopy(kwargs)
        
        for k in self.keys:
            img = mod_kwargs[k]

            # Get the center crop
            c, h, w  = img.shape
            assert (h >= self.height) and (w >= self.width)

            top = (h - self.height) // 2
            left = (w - self.width) // 2
            bottom = top + self.height
            right = left + self.width

            img = img[:, top:bottom, left:right]

            # Make sure the image is the right size wit han assert and if there is an error, print the shapes
            assert (img.shape[0] == kwargs[k].shape[0]) and (img.shape[1] == self.height) and (img.shape[2] == self.width), \
                    f'img.shape: {img.shape}, self.height: {self.height}, self.width: {self.width}'

            mod_kwargs[k] = img

        return mod_kwargs





