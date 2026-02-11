import numpy as np

from albumentations.core.transforms_interface import BasicTransform

# create a transform that creates a 3-channel image from a 1-channel image
class SingleChannelToRGB(BasicTransform):
    def __init__(self, keys, always_apply=False, p=1.0):
        super(SingleChannelToRGB, self).__init__(always_apply, p)

        self.keys = keys
        self.p = p
        return

    def __call__(self, *args, force_apply=False, **kwargs):

        for key in self.keys:
            image = kwargs[key].copy()

            image = np.concatenate((image, image, image), axis=0)

            kwargs[key] = image

        return kwargs