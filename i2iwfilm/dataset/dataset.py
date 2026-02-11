import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

import albumentations
import collections

from hydra.utils import call, instantiate
from functools import partial

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class ImagePaths(data.Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        # preprocessing pipeline: 
        # if image has different size than self.size, rescale it to self.size
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        # crop the center of the image
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        #this makes sure that the image ha correct shape
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        # open the image
        image = Image.open(image_path)
        # convert it to RGB
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # convert it to numpy array
        image = np.array(image).astype(np.uint8)
        # apply preprocessing
        image = self.preprocessor(image=image)["image"]
        # make sur that the image is in range [-1, 1] + convert it to float32
        image = (image / 127.5 - 1.0).astype(np.float32)
        # W x H x C -> C x W x H for pytorch
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


class NpzImagePaths(ImagePaths):
    def __init__(self, root_dir, 
                 partition, 
                 input_dataset_file,
                 cond_dataset_file,
                 input_resolution,
                 transforms=None,
                 no_preprocessing=False,
                 angle_map_dataset=None,
                size=None):
        # We don't need to call constructor of ImagePaths.
        # super().__init__(root_dir, size)
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = albumentations.Compose(transforms_init)
        self.transforms = transforms

        self.root_dir = root_dir
        self.partition = partition
        self.input_dataset_file = input_dataset_file
        self.cond_dataset_file = cond_dataset_file
        self.input_resolution = input_resolution

        self.angle_map_dataset = angle_map_dataset

        self.no_preprocessing = no_preprocessing

        #if root_dir is not an absolute path, add the SUNTM_PATH to it
        if not os.path.isabs(root_dir):
            print(f'ROOT_DIR:{root_dir} is not an absolute path, adding SUNTM_PATH to it')
            self.root_dir = os.path.join(os.environ["SUNTM_PATH"], root_dir)
        
        print(f'---------------------  DATASET  ---------------------')
        print(f'ROOT_DIR:{self.root_dir}')
        print(f'INPUT_DATASET_FILE:{input_dataset_file}')
        print(f'COND_DATASET_FILE:{cond_dataset_file}')
        if self.angle_map_dataset is not None:
            print(f'ANGLE_MAP_DATASET:{angle_map_dataset}')
        print(f'INPUT_RESOLUTION:{input_resolution}')
        print(f'-----------------------------------------------------')

        # read the dataset file, each line is a path to a .npz file
        input_files = []
        with open(os.path.join(self.root_dir, input_dataset_file), "r") as f:
            paths = f.read().splitlines()
            paths = [os.path.join(root_dir, path) for path in paths]
            input_files.extend(paths)


        # read the condition dataset file, each line is a path to a .npz file
        condition_files = []
        with open(os.path.join(self.root_dir, cond_dataset_file), "r") as f:
            paths = f.read().splitlines()
            paths = [os.path.join(root_dir, path) for path in paths]
            condition_files.extend(paths)

       
        if self.angle_map_dataset is None:

            self.joint_files = list(zip(input_files, condition_files))

        else:
            # read the angle map dataset file, each line is a path to a .npz file
            angle_map_files = []
            with open(os.path.join(self.root_dir, angle_map_dataset), "r") as f:
                paths = f.read().splitlines()
                paths = [os.path.join(root_dir, path) for path in paths]
                angle_map_files.extend(paths)

            self.joint_files = list(zip(input_files, condition_files, angle_map_files))

        self.files = self.joint_files.copy()
            

        # preprocessing pipeline: 
        # if image has different size than self.size, rescale it to self.size
        import cv2
        self.rescaler = albumentations.Resize(height=self.input_resolution,
                                              width=self.input_resolution,
                                              interpolation=cv2.INTER_NEAREST)
        # self.rescaler = albumentations.SmallestMaxSize(max_size=self.input_resolution)
        # crop the center of the image
        self.cropper = albumentations.CenterCrop(height=self.input_resolution, width=self.input_resolution)
        #this makes sure that the image ha correct shape
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def preprocess_image(self, image_path):
        # Original full-disc image has been normalized to 0,1 BEFORE CROPPING and dumped as a .npz file
        # The cropped image must be processed to be in the range [-1,1] in order to be fed to a diffusion model:
        #    The loaded image must thus be multiplied by 2 and subtracted by 1
        image = np.load(image_path)["arr_0"]  # 1 x N x M

        if image.shape[1] != self.input_resolution or image.shape[2] != self.input_resolution:
            if not self.no_preprocessing:
                image = self.preprocessor(image=image)["image"]

        # if shape is missing a dimension, add it
        if len(image.shape) == 2:
              image = np.expand_dims(image, axis=0)

        image = image.astype(np.float32)
        
        #  Normalize to -1,1
        # image = (image*2.0 - 1.0).astype(np.float32)
        # # clip to -1,1
        # image = np.clip(image, -1.0, 1.0)

        # assert that image is in float32
        assert image.dtype == np.float32
        return image
    
        
    def __len__(self):
        return len(self.files)
    
    def  __getitem__(self, index: int, do_transform=True):
        i = index % len(self.joint_files)
        sample = {}

        path = self.files[i][0]
        cond_path = self.files[i][1]
        
        sample['gt_image'] = self.preprocess_image(self.files[i][0])
        sample['cond_image'] = self.preprocess_image(self.files[i][1])
        sample['path'] = os.path.basename(path)
        sample['cond_path'] = os.path.basename(cond_path)

        if self.angle_map_dataset is not None:
            sample['angle_map'] = self.preprocess_image(self.files[i][2])
            sample['angle_map_path'] = os.path.basename(self.files[i][2])

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        return sample

class aia2hmi_NpzImagePathsV3(ImagePaths):
    def __init__(self, root_dir, 
                 partition, 
                 input_dataset_file,
                 cond_dataset_file,
                 input_resolution,
                 transforms=None,
                 no_preprocessing=False,
                size=None):
        # We don't need to call constructor of ImagePaths.
        # super().__init__(root_dir, size)
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = albumentations.Compose(transforms_init)
        self.transforms = transforms

        self.root_dir = root_dir
        self.partition = partition
        self.input_dataset_file = input_dataset_file
        self.cond_dataset_file = cond_dataset_file
        self.input_resolution = input_resolution

        self.no_preprocessing = no_preprocessing

        #if root_dir is not an absolute path, add the SUNTM_PATH to it
        if not os.path.isabs(root_dir):
            print(f'ROOT_DIR:{root_dir} is not an absolute path, adding SUNTM_PATH to it')
            self.root_dir = os.path.join(os.environ["SUNTM_PATH"], root_dir)
        
        print(f'---------------------  DATASET  ---------------------')
        print(f'ROOT_DIR:{self.root_dir}')
        print(f'INPUT_DATASET_FILE:{input_dataset_file}')
        print(f'COND_DATASET_FILE:{cond_dataset_file}')
        print(f'INPUT_RESOLUTION:{input_resolution}')
        print(f'-----------------------------------------------------')

        # read the dataset file, each line is a path to a .npz file
        input_files = []
        with open(os.path.join(self.root_dir, input_dataset_file), "r") as f:
            paths = f.read().splitlines()
            paths = [os.path.join(root_dir, path) for path in paths]
            input_files.extend(paths)


        # read the condition dataset file, each line is a path to a .npz file
        condition_files = []
        with open(os.path.join(self.root_dir, cond_dataset_file), "r") as f:
            paths = f.read().splitlines()
            paths = [os.path.join(root_dir, path) for path in paths]
            condition_files.extend(paths)

        self.joint_files = list(zip(input_files, condition_files))
       
        self.files = self.joint_files.copy()
            

        # preprocessing pipeline: 
        # if image has different size than self.size, rescale it to self.size
        import cv2
        self.rescaler = albumentations.Resize(height=self.input_resolution,
                                              width=self.input_resolution,
                                              interpolation=cv2.INTER_NEAREST)
        # self.rescaler = albumentations.SmallestMaxSize(max_size=self.input_resolution)
        # crop the center of the image
        self.cropper = albumentations.CenterCrop(height=self.input_resolution, width=self.input_resolution)
        #this makes sure that the image ha correct shape
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def preprocess_image(self, image_path, type="aia"):
        # Original full-disc image has been normalized to 0,1 BEFORE CROPPING and dumped as a .npz file
        # The cropped image must be processed to be in the range [-1,1] in order to be fed to a diffusion model:
        #    The loaded image must thus be multiplied by 2 and subtracted by 1
        image = np.load(image_path)[type]  # 1 x N x M

        if image.shape[1] != self.input_resolution or image.shape[2] != self.input_resolution:
            if not self.no_preprocessing:
                image = self.preprocessor(image=image)["image"]

        # if shape is missing a dimension, add it
        if len(image.shape) == 2:
              image = np.expand_dims(image, axis=0)

        image = image.astype(np.float32)
        
        #  Normalize to -1,1
        # image = (image*2.0 - 1.0).astype(np.float32)
        # # clip to -1,1
        # image = np.clip(image, -1.0, 1.0)

        # assert that image is in float32
        assert image.dtype == np.float32
        return image
    
        
    def __len__(self):
        return len(self.files)
    
    def  __getitem__(self, index: int, do_transform=True):
        i = index % len(self.joint_files)
        sample = {}

        path = self.files[i][0]
        cond_path = self.files[i][1]
        
        sample['gt_image'] = self.preprocess_image(self.files[i][0], type="hmi")
        sample['cond_image'] = self.preprocess_image(self.files[i][1], type="aia")
        sample['path'] = os.path.basename(path)
        sample['cond_path'] = os.path.basename(cond_path)

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        return sample









