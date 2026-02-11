import os
from PIL import Image

import torch
import torchvision
import torchvision.utils as vutils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import numpy as np

import wandb
import matplotlib.pyplot as plt

class I2IwFiLM_TestVizualizeImageTranslationCallback(Callback):
    def __init__(self, batch_freq, output_dir, max_images, log_steps=[], clamp=True, log_wandb=False):
        super().__init__()

        self.output_dir = output_dir
        self.batch_freq = batch_freq
        self.max_images = max_images
        self.log_steps = log_steps

        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }

        self.clamp = clamp

        self.log_wandb = log_wandb

        str_tmp = f"output_dir: {output_dir}, batch_freq: {batch_freq}, max_images: {max_images}, log_steps: {log_steps}, clamp: {clamp}"

    def _wandb(self, pl_module, images, global_step, split, batch_idx):
        keys = list(images.keys())

        bs = self.max_images

        for i in range(bs):
            # 1) concatentate values of all keys in a row: [bs, H, W * len(keys)]
            row = torch.cat([images[k][i] for k in keys], dim=-1)
            
            # 2) rescale to [0, 1]
            row_png = (row + 1.) / 2.

            # 3) rescale to [0, 255]
            row_png = row_png * 255.
            
            # 4) convert to numpy array and to uint8
            row_png = row_png.permute(1, 2, 0).numpy().astype(np.uint8)
            row_png = np.squeeze(row_png)

            # 5) save image
            img_idx = batch_idx * bs + i
            img_name = f"{split}_{img_idx}.png"

            self.img_to_log[img_name] = row_png


    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False
    
    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        assert dump_type in ['source', 'target', 'sample']
        # 1) Output of VQGAN decoder should be in [-1, 1] 

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image_png = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image_png = np.squeeze(image_png)

        image_npy = image.permute(1, 2, 0).numpy().astype(np.float32)
        image_npy = np.squeeze(image_npy)


        return image_png, image_npy


    
    @rank_zero_only
    def log_local(self, save_dir, split, outputs,
                  global_step, current_epoch, batch_idx):
        keys = list(outputs.keys())

        bs = outputs[keys[0]].shape[0]

        os.makedirs(os.path.join(save_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "npz"), exist_ok=True)

        inputs = outputs["Input"].clone().cpu()
        targets = outputs["Target"].clone().cpu()
        samples = outputs["Reconstruction"].clone().cpu()
        names = outputs["name"]
        
        for i in range(bs):
            img_idx = batch_idx * bs + i

            prefix_name = names[i]

            src_png, src_npz  = self.to_dump_format(inputs[i], split, batch_idx, bs, dump_type="source")
            
            src_name = f"{prefix_name}_source.png"
            src_path = os.path.join(save_dir, "png", src_name)

            out_png, out_npz = self.to_dump_format(samples[i], split, batch_idx, bs, dump_type="sample")
            out_name = f"{prefix_name}_sample.png"
            out_path = os.path.join(save_dir, "png", out_name)

            tgt_png, tgt_npz = self.to_dump_format(targets[i], split, batch_idx, bs, dump_type="target")
            tgt_name = f"{prefix_name}_target.png"
            tgt_path = os.path.join(save_dir, "png", tgt_name)

            self.img_to_log[src_name] = src_png
            self.img_to_log[tgt_name] = tgt_png
            self.img_to_log[out_name] = out_png

            Image.fromarray(src_png, mode='L').save(src_path)
            Image.fromarray(out_png, mode='L').save(out_path)
            Image.fromarray(tgt_png, mode='L').save(tgt_path)

            np.savez_compressed(os.path.join(save_dir, "npz", f"{prefix_name}_source.npz"), src_npz)
            np.savez_compressed(os.path.join(save_dir, "npz", f"{prefix_name}_sample.npz"), out_npz)
            np.savez_compressed(os.path.join(save_dir, "npz", f"{prefix_name}_target.npz"), tgt_npz)





    def log_img(self, pl_module, batch, outputs, batch_idx, split="train"):
        
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            outputs["name"] = batch["path"]
            
            self.log_local(self.output_dir, split, outputs,
                            pl_module.global_step, pl_module.current_epoch, batch_idx)
        
            if is_train:
                pl_module.train()

    def on_test_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = {}

    def on_test_epoch_end(self, trainer, pl_module):

        if trainer.logger is None:
            return
        
        if self.log_wandb:
            
            # create wandb table
            table = wandb.Table(columns=["image"])
            for i, (img_name, img) in enumerate(self.img_to_log.items()):
                table.add_data(wandb.Image(img, caption=img_name))

            # log table
            trainer.logger.experiment.log({"test_reconstruction": table})

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, outputs["outputs"], batch_idx, split="test")


