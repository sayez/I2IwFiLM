import numpy as np

from copy import deepcopy

from albumentations.core.transforms_interface import BasicTransform

import matplotlib.pyplot as plt

class CenteredNormalScaling(BasicTransform):
    '''
        Rescale image to [-1,1] from [0,255]
    '''
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        
        image = kwargs['image'].copy()

        scaled_image = (image/127.5 - 1.0).astype(np.float32)

        kwargs['image'] = scaled_image

        return kwargs
    
class CenteredNormalScalingKeys(BasicTransform):
    '''
        Rescale image to [-1,1] from [0,255]
    '''
    def __init__(self, keys=[], always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.keys = keys

    def __call__(self, *args, force_apply=False, **kwargs):

        for key in self.keys:
            image = kwargs[key].copy()

            scaled_image = (image/127.5 - 1.0).astype(np.float32)
            # clip to -1,1
            scaled_image = np.clip(scaled_image, -1.0, 1.0)

            kwargs[key] = scaled_image

        return kwargs
    
class CenteredNormalScalingKeysMinMaxClip(BasicTransform):
    '''
        Rescale image to [-1,1] from [min_value,max_value]
    '''
    def __init__(self, keys=[], min_val=0, max_val=255, clip=True, always_apply=False, p=1, show=False):
        super().__init__(always_apply, p)
        self.keys = keys
        self.min_value = min_val
        self.max_value = max_val

        self.clip = clip
        self.show = show

    def __call__(self, *args, force_apply=False, **kwargs):
        original_kwargs = deepcopy(kwargs)
        mod_kwargs = deepcopy(kwargs)

        for key in self.keys:
            image = kwargs[key].copy()

            scaled_image = (image - self.min_value) / (self.max_value - self.min_value) # normalize to [0,1] if needed
            scaled_image = 2 * scaled_image - 1 # scale to [-1,1]

            if self.clip:
                # clip to -1,1
                scaled_image = np.clip(scaled_image, -1.0, 1.0)

            mod_kwargs[key] = scaled_image

        if self.show:
            fig , ax = plt.subplots(len(self.keys),2, figsize=(8,8))
            for i, k in enumerate(self.keys):
                ax[i,0].imshow(original_kwargs[k].squeeze(), cmap='gray', interpolation='none', vmin=self.min_value, vmax=self.max_value)
                ax[i,0].set_title('Original')
                ax[i,0].set_xlabel(f'Pixel Value [{self.min_value},{self.max_value}]:\n {original_kwargs[k].min():4f}, {original_kwargs[k].max():4f}')

                ax[i,1].imshow(mod_kwargs[k].squeeze(), cmap='gray', interpolation='none', vmin=-1, vmax=1)
                ax[i,1].set_title('Scaled')
                ax[i,1].set_xlabel(f'Pixel Value [-1,1]:\n {mod_kwargs[k].min():4f}, {mod_kwargs[k].max():4f}')
            fig.suptitle('Centered Normal Scaling')
            fig.tight_layout()
            plt.show()

        return mod_kwargs
    
class IdentityTransform(BasicTransform):
    '''
        Apply identity transform to image.
        Output is same as input.
        Used as placeholder in the transform list of configuration file: 
            to be replaced by other transforms in training script.
    '''
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        return kwargs
    


class LogarithmTransform(BasicTransform):
    '''
        Apply logarithm to image:
        Enhances contrast in the darker regions of the image
    '''
    def __init__(self, input_max_value=255, output_max_value=255, always_apply=False, p=1):
        super().__init__(always_apply, p)

        self.input_max_value = input_max_value
        self.output_max_value = output_max_value


    def __call__(self, *args, force_apply=False, **kwargs):
            
        image = kwargs['image'].copy()

        c = (self.output_max_value) / np.log(self.input_max_value + 1 )

        image_log = c * np.log(image + 1)

        kwargs['image'] = image_log

        return kwargs
    

class LogarithmTransformKeys(BasicTransform):
    '''
        Apply logarithm to image:
        Enhances contrast in the darker regions of the image
    '''
    def __init__(self, keys=[], input_max_value=255, output_max_value=255, always_apply=False, p=1):
        super().__init__(always_apply, p)

        self.keys = keys
        self.input_max_value = input_max_value
        self.output_max_value = output_max_value


    def __call__(self, *args, force_apply=False, **kwargs):
            
        for key in self.keys:
            image = kwargs[key].copy()

            c = (self.output_max_value) / np.log(self.input_max_value + 1 )

            image_log = c * np.log(image + 1)

            kwargs[key] = image_log
        
        return kwargs
    
class ExponentialTransform(BasicTransform):
    '''
        Apply exponential to image:
        Enhances contrast in the lighter regions of the image
    '''
    def __init__(self, input_max_value, output_max_value, always_apply=False, p=1):
        super().__init__(always_apply, p)

        self.input_max_value = input_max_value
        self.output_max_value = output_max_value

    def __call__(self, *args, force_apply=False, **kwargs):
                
        image = kwargs['image'].copy()

        c = (self.output_max_value) / np.log(self.input_max_value + 1 )

        image_exp = np.exp(image)**(1/c) - 1

        kwargs['image'] = image_exp
        
        return kwargs
    
class ExponentialTransformKeys(BasicTransform):
    '''
        Apply exponential to image:
        Enhances contrast in the lighter regions of the image
    '''
    def __init__(self, keys=[], input_max_value=255, output_max_value=255, always_apply=False, p=1):
        super().__init__(always_apply, p)

        self.keys = keys
        self.input_max_value = input_max_value
        self.output_max_value = output_max_value

    def __call__(self, *args, force_apply=False, **kwargs):
                
        for key in self.keys:
            image = kwargs[key].copy()

            c = (self.output_max_value) / np.log(self.input_max_value + 1 )

            image_exp = np.exp(image)**(1/c) - 1

            kwargs[key] = image_exp
        
        return kwargs


        

     
            
     

