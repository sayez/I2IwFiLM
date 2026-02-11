import torch
import torch.nn as nn
import numpy as np
import cv2


from copy import deepcopy

from albumentations.core.transforms_interface import BasicTransform
from albumentations.augmentations.geometric.transforms import Flip

class RandomFlipDictKeys(Flip):
    def __init__(self, keys, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.keys = keys

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        for key in self.keys:
            kwargs[key] = processed_kwargs['image']
            
        return kwargs
    
class RandomFlipDictKeys2(BasicTransform):
    def __init__(self, keys, flip_v, flip_h, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.keys = keys
        self.flip_v = flip_v
        self.flip_h = flip_h

        self.possible_d = []
        if self.flip_v:
            self.possible_d.append('v')
        if self.flip_h:
            self.possible_d.append('h')
        if self.flip_v and self.flip_h:
            self.possible_d.append('hv')

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)

        if (torch.rand(1) < self.p) or self.always_apply or force_apply:
            # select one of the possible flips randomly
            d = self.possible_d[torch.randint(len(self.possible_d), (1,))]

            if d == 'v':
                for k in self.keys:
                    # numpy version
                    mod_kwargs[k] = np.flip(kwargs[k], axis=(1,)).copy()
            elif d == 'h':
                for k in self.keys:
                    # numpy version
                    mod_kwargs[k] = np.flip(kwargs[k], axis=(2,)).copy()
            elif d == 'hv':
                for k in self.keys:
                    # numpy version
                    mod_kwargs[k] = np.flip(kwargs[k], axis=(1,2)).copy()

            for key in self.keys:
                kwargs[key] = mod_kwargs[key]
        else:
            pass

        return kwargs
    
class RandomRotate90DictKeys(BasicTransform):
    def __init__(self, keys, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.keys = keys

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)

        if (torch.rand(1) < self.p) or self.always_apply or force_apply:
            # select one of the possible rotations randomly
            cur_rot = torch.randint(4, (1,)).item()
            for k in self.keys:
                # numpy version
                mod_kwargs[k] = np.rot90(kwargs[k], k=cur_rot, axes=(1,2)).copy()
            
            for key in self.keys:
                kwargs[key] = mod_kwargs[key]
        else:
            pass

        return kwargs

import matplotlib.pyplot as plt

def rotate_CV_bound(image, angle, interpolation):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),flags=interpolation)

class RandomRotateDictKeys(BasicTransform):
    def __init__(self, keys, limit=90, show=False, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.keys = keys
        self.limit = limit

        self.show = show

    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        if (torch.rand(1) < self.p) or self.always_apply or force_apply:
            # select one of the possible rotations randomly
            cur_rot = torch.randint(-self.limit, self.limit, (1,)).item()
            cols, rows = kwargs[self.keys[0]].shape[-1], kwargs[self.keys[0]].shape[-2]
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-cur_rot,1) 
            # print(f'cur_rot: {cur_rot}')
            for k in self.keys:
                # numpy version
                # CxHxW -> HxWxC
                tmp = np.moveaxis(mod_kwargs[k], 0, -1)
                tmp = cv2.warpAffine(tmp, M, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=(-1))
                tmp = tmp[..., np.newaxis]
                # HxWxC -> CxHxW
                mod_kwargs[k] = np.moveaxis(tmp, -1, 0)

            if self.show:
                fig , ax = plt.subplots(len(self.keys), 2, figsize=(5,5))
                for i, k in enumerate(self.keys):
                    ax[i,0].imshow(original_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                    ax[i,0].set_title(original_kwargs["path"])
                    ax[i,1].imshow(mod_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                fig.suptitle(f'RandomRotateDictKeys: {cur_rot}')
                fig.tight_layout()
                plt.show()

            for key in self.keys:
                kwargs[key] = mod_kwargs[key]

            
        else:
            pass

        return kwargs

class RandomCentralCropDictKeys(BasicTransform):
    def __init__(self, keys, crop_size, x=128, y=128, bw=256, bh=256, show=False , always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.keys = keys
        self.crop_size = crop_size
        self.x = x
        self.y = y
        self.bw = bw
        self.bh = bh

        self.show = show


    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        if (torch.rand(1) < self.p) or self.always_apply or force_apply:
            x = mod_kwargs[self.keys[0]].shape[2] // 2 - self.x
            y = mod_kwargs[self.keys[0]].shape[1] // 2 - self.y
            w, h = self.crop_size, self.crop_size
            # sample the centre of the crop on original image
            sample_x = np.random.randint(x, x + self.bw + 1)
            sample_y = np.random.randint(y, y + self.bh + 1)
            # print(f'RandomCentralCropDictKeys: {sample_x}, {sample_y}')

            # for each key, extract a crop around the sampled centre
            for k in self.keys:
                # print(f'RandomCentralCropDictKeys: {k}, {mod_kwargs[k].shape}')
                min_x = max(0, sample_x - (w // 2))
                min_y = max(0, sample_y - (h // 2))
                max_x = min(mod_kwargs[k].shape[2], sample_x + (w // 2))
                max_y = min(mod_kwargs[k].shape[1], sample_y + (h // 2))
                
                mod_kwargs[k] = mod_kwargs[k][:, min_y:max_y, min_x:max_x]
                
                assert mod_kwargs[k].shape[1] == self.crop_size
                assert mod_kwargs[k].shape[2] == self.crop_size


            if self.show:
                fig , ax = plt.subplots(len(self.keys),2, figsize=(5,5))
                for i, k in enumerate(self.keys):
                    ax[i,0].imshow(kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                    ax[i,0].set_title(original_kwargs["path"])
                    ax[i,1].imshow(mod_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                fig.suptitle(f'RandomCentralCropDictKeys:')
                fig.tight_layout()
                plt.show()
                

            for key in self.keys:
                kwargs[key] = mod_kwargs[key]
        else:
            pass

        return kwargs

from albumentations.augmentations.transforms import RandomBrightnessContrast
from omegaconf import ListConfig
class RandomBrightnessContrastDictKeys(BasicTransform):
    def __init__(self, keys, brightness_limit=0.2, contrast_limit=0.2, 
                 brightness_by_max=True,
                  show=False, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.keys = keys
        self.brightness_limit = brightness_limit # may be a list of strings, convert to float
        
        if isinstance(self.brightness_limit, list) or isinstance(self.brightness_limit, ListConfig):
            
            self.brightness_limit = [float(x) for x in self.brightness_limit]
            
            if type(self.brightness_limit[0])== str and self.brightness_limit[0][0] == '-':
                self.brightness_limit[0] = -float(self.brightness_limit[0][1:])
            
            self.brightness_limit = tuple(self.brightness_limit)
            
        self.contrast_limit = contrast_limit # may be a list of strings, convert to float
        if isinstance(self.contrast_limit, list):
            self.contrast_limit = [float(x) for x in self.contrast_limit]
            if type(self.contrast_limit[0])== str and self.contrast_limit[0][0] == '-':
                self.contrast_limit[0] = -float(self.contrast_limit[0][1:])
            self.contrast_limit = tuple(self.contrast_limit)

        self.brightness_by_max = brightness_by_max

        self.show = show

    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        if (torch.rand(1) < self.p) or self.always_apply or force_apply:

            for k in self.keys:
                transform = RandomBrightnessContrast(brightness_limit=self.brightness_limit, contrast_limit=self.contrast_limit, 
                                                     brightness_by_max=self.brightness_by_max, p=1.0)
                
                mod_kwargs[k] = transform(image=mod_kwargs[k])['image']

            if self.show:
                fig , ax = plt.subplots(len(self.keys),2, figsize=(5,5))
                for i, k in enumerate(self.keys):
                    ax[i,0].imshow(kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                    ax[i,0].set_title(original_kwargs["path"])
                    ax[i,1].imshow(mod_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                fig.suptitle(f'RandomBrightnessContrastDictKeys:')
                fig.tight_layout()
                plt.show()
                
            for key in self.keys:
                kwargs[key] = mod_kwargs[key]
        else:
            
            pass

        return kwargs

class ResizeDictKeys(BasicTransform):
    def __init__(self, keys, height, width,show=False, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.keys = keys
        self.height = height
        self.width = width

        self.show = show

    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        for k in self.keys:
            # numpy version
            tmp = np.moveaxis(mod_kwargs[k], 0, -1)
            tmp = cv2.resize(tmp, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            tmp = tmp[..., np.newaxis]
            mod_kwargs[k] = np.moveaxis(tmp, -1, 0)
            

        if self.show:
            fig , ax = plt.subplots(len(self.keys),2, figsize=(5,5))
            for i, k in enumerate(self.keys):
                ax[i,0].imshow(kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                ax[i,0].set_title(original_kwargs["path"])
                ax[i,1].imshow(mod_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
            fig.suptitle(f'ResizeDictKeys:')
            fig.tight_layout()
            plt.show()
            
        for key in self.keys:
            kwargs[key] = mod_kwargs[key]

        return kwargs
    
class ClipDictKeys(BasicTransform):
    def __init__(self, keys, min_val=-1.0, max_val=1.0, show=False, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.keys = keys
        self.min_val = min_val
        self.max_val = max_val

        self.show = show

    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        for k in self.keys:
            # numpy version
            mod_kwargs[k] = np.clip(mod_kwargs[k], self.min_val, self.max_val)

        if self.show:
            fig , ax = plt.subplots(len(self.keys),2, figsize=(5,5))
            for i, k in enumerate(self.keys):
                ax[i,0].imshow(kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
                ax[i,0].set_title(original_kwargs["path"])
                ax[i,1].imshow(mod_kwargs[k][0], cmap='gray', vmin=-1, vmax=1)
            fig.suptitle(f'ClipDictKeys:')
            fig.tight_layout()
            plt.show()
            
        for key in self.keys:
            kwargs[key] = mod_kwargs[key]

        return kwargs

