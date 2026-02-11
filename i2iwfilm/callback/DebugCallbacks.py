import os
from typing import Any, Optional
from PIL import Image
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torchvision
import torchvision.utils as vutils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt

import numpy as np



import wandb

from i2iwfilm.config import logger


class TestVisualizeInputsOutputs(Callback):
    def __init__(self, output_dir, inline=False, batch_freq=1, max_images=32):
        super().__init__()

        print("INITIALIZING TestVisualizeInputsOutputs")

        self.output_dir = output_dir
        self.inline = inline
        self.batch_freq = batch_freq
        self.max_images = max_images

    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        bs = None
        img_keys = []
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor):
                bs = batch[k].shape[0]
                img_keys.append(k)

        if self.inline:
            # create a grid of images
            fig, axes = plt.subplots(bs, len(img_keys), figsize=(len(img_keys)*3, bs*3))
            for img_k in range(len(img_keys)):
                for b in range(bs):
                    axes[b, img_k].imshow(batch[img_keys[img_k]][b].permute(1,2,0).cpu().numpy(), cmap="gray")
                    axes[b, img_k].set_title(f"{img_keys[img_k]}")
                    axes[b, img_k].axis("off")

        return 
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return


class VisualizePredictions(Callback):
    def __init__(self, output_dir, inline=True, log_wandb=False, batch_freq=1, max_images=32):
        super().__init__()

        print("INITIALIZING VisualizePredictions")

        self.output_dir = output_dir
        self.inline = inline
        self.batch_freq = batch_freq
        self.max_images = max_images

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x_start = batch['gt_image']
        t = outputs['timesteps']
        x_t = outputs['noisy']
        predictions = outputs['predictions']

        # show a figure with x_start and x_t for debugging
        bs = x_start.shape[0]
        num_rows = 2 if predictions is None else 3
        fig, ax = plt.subplots(num_rows, bs, figsize=(num_rows*4, bs*2.5))

        tmp_start = x_start[:, 0, :, :].detach().cpu().numpy()
        # Get the min and max of each image (two last dimensions)
        min_start = tmp_start.min(axis=(1,2), keepdims=True).squeeze()
        max_start = tmp_start.max(axis=(1,2), keepdims=True).squeeze()
        min_max_start = np.vstack([min_start, max_start])

        tmp_end = x_t[:, 0, :, :].detach().cpu().numpy()
        # Get the min and max of each image (two last dimensions)
        min_end = tmp_end.min(axis=(1,2), keepdims=True).squeeze()
        max_end = tmp_end.max(axis=(1,2), keepdims=True).squeeze()
        min_max_end = np.vstack([min_end, max_end])

        for i in range(bs):

            ax[0,i].imshow(tmp_start[i], cmap='gray', vmin=-1, vmax=1)
            ax[1,i].imshow(tmp_end[i], cmap='gray') 
            ax[0,i].axis('off')
            ax[1,i].axis('off')
            ax[0,i].set_title('x_start')
            ax[1,i].set_title('timestep ={}'.format(t[i].detach().cpu().numpy()))
        
        if predictions is not None:
            tmp_pred = predictions[:, 0, :, :].detach().cpu().numpy()
            # Get the min and max of each image (two last dimensions)
            min_pred = tmp_pred.min(axis=(1,2), keepdims=True).squeeze()
            max_pred = tmp_pred.max(axis=(1,2), keepdims=True).squeeze()
            min_max_pred = np.vstack([min_pred, max_pred])

            for i in range(bs):
                ax[2,i].imshow(tmp_pred[i], cmap='gray')
                ax[2,i].axis('off')
                ax[2,i].set_title('predicted x_start')
        fig.tight_layout()
        plt.show()



class ValidationVizualizeReconstructionCallback(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        assert dump_type in ["condition", 'source', 'target', 'sample']
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)

        return image
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        wandb.define_metric("epoch")
    
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    # def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        assert not pl_module.training
        
        x = batch['gt_image']
        x_cond = batch['cond_image']

        device = x.device    

        # create pure noise starting points
        noise = torch.randn_like(x)

        # create the dict with condition tensor
        model_kwargs = dict(condition=x_cond)

        # Sample images:
        samples = pl_module.diffusion.p_sample_loop(
            pl_module.model, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

        conditions = x_cond.clone().cpu()
        inputs = noise.clone().cpu()
        targets = x.clone().cpu()
        reconstruction = samples.clone().cpu()

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = inputs.shape[0]
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cond = self.to_dump_format(conditions[i], split, batch_idx, bs, dump_type="condition")
            src  = self.to_dump_format(inputs[i], split, batch_idx, bs, dump_type="source")
            out = self.to_dump_format(reconstruction[i], split, batch_idx, bs, dump_type="sample") 
            tgt = self.to_dump_format(targets[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                    wandb.Image(cond), wandb.Image(src),
                                    wandb.Image(out), wandb.Image(tgt)))


    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        logger = pl_module.logger
        if logger is None:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                
                table = wandb.Table(columns=["epoch", "batch", "id", "condition", "source", "sample", "target"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, cond, src, out, tgt = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(cond), wandb.Image(src),
                                        wandb.Image(out), wandb.Image(tgt))

                # print("Logging to remote wandb...")
                trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

            self.img_to_log = []


class LBBDM_ValidationVizualizeReconstructionCallback(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        assert dump_type in ["condition", 'source', 'target', 'sample']
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)
        # print(f"row shape: {row_png.shape}, dtype: {row_png.dtype}, min: {row_png.min()}, max: {row_png.max()}")

        return image
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        wandb.define_metric("epoch")
    
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        assert not pl_module.training

        logger = pl_module.logger
        if logger is None:
            return
        
        x = outputs["Target"]
        x_cond = outputs["Input"]

        samples = outputs["Reconstruction"]

        conditions = x_cond.clone().cpu()
        targets = x.clone().cpu()
        reconstruction = samples.clone().cpu()

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = targets.shape[0]
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cond = self.to_dump_format(conditions[i], split, batch_idx, bs, dump_type="condition")
            out = self.to_dump_format(reconstruction[i], split, batch_idx, bs, dump_type="sample") 
            tgt = self.to_dump_format(targets[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                    wandb.Image(cond),
                                    wandb.Image(out), wandb.Image(tgt)))


    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        logger = pl_module.logger
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):            
                table = wandb.Table(columns=["epoch", "batch", "id", "condition", "sample", "target"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, cond, out, tgt = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(cond),
                                        wandb.Image(out), wandb.Image(tgt))

                # print("Logging to remote wandb...")
                trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

            self.img_to_log = []

class I2IwFiLM_S1_Validation_VisualizeReconstructionCallback(Callback):
    def __init__(self, output_dir, inline=False, batch_freq=1, max_images=32):
        super().__init__()

        print("INITIALIZING I2IwFiLM_S1_VisualizeReconstructionCallback")

        self.output_dir = output_dir
        self.inline = inline
        self.batch_freq = batch_freq
        self.max_images = max_images

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        bs = None
        img_keys = []

        results = outputs["outputs"]
        bs = results["Input"].shape[0]
        for k in list(results.keys()):
            if isinstance(results[k], torch.Tensor):
                img_keys.append(k)
                
        if self.inline:

            # create a grid of images
            fig, axes = plt.subplots(bs, len(img_keys), figsize=(len(img_keys)*3, bs*3))
            for img_k in range(len(img_keys)):
                for b in range(bs):
                    axes[b, img_k].imshow(results[img_keys[img_k]][b].permute(1,2,0).cpu().numpy(), cmap="gray")
                    axes[b, img_k].set_title(f"{img_keys[img_k]}")
                    axes[b, img_k].axis("off")


        else:
            # save images to disk
            for img_k in img_keys:
                for b in range(bs):
                    img = results[img_k][b].permute(1,2,0).cpu().numpy()
                    img = (img + 1.) / 2.
                    img = (img * 255.).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(f"{self.output_dir}/{img_k}_{batch_idx}_{b}.png")

        return
        

class I2IwFiLM_S1_Validation_Log_Reconstructions(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.first_log = True

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)

        return image
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC epoch')
        wandb.define_metric("epoch")
    
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        assert not pl_module.training
        
        tgt = batch['gt_image'] # Target image
        inp = batch['cond_image'] # Conditional image (input)

        batch_size = tgt.shape[0]

        reconstructions = outputs["outputs"]["Reconstruction"]
        
        # get all tensors to cpu
        reconstructions = reconstructions.cpu()
        tgt = tgt.cpu()
        inp = inp.cpu()

        ####

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")
            cur_out = self.to_dump_format(reconstructions[i], split, batch_idx, bs, dump_type="sample") 
            cur_tgt = self.to_dump_format(tgt[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                    wandb.Image(cur_inp),
                                    wandb.Image(cur_out), wandb.Image(cur_tgt)))


    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.log_wandb:

            logger = pl_module.logger
            if logger is None:
                return
            if not isinstance(logger, WandbLogger):
                return
        
            if not self.first_log:
                api = wandb.Api()
                run_id = wandb.run.id
                run_project = wandb.run.project

                self.run_id = run_id
                self.run_project = run_project
            
                latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-validation_reconstruction:latest')
                if latest_artif.file_count == 1:
                    latest_artif.delete(delete_aliases=True)
                    artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                    if len(artifs) > 0:
                        artifs[0].aliases += ["latest"]
                        artifs[0].save()

            table = wandb.Table(columns=["epoch", "batch", "id", "Input", "Reconstruction", "Target"])
            
            for i, d in enumerate(self.img_to_log):
                epoch_num, batch_idx, sample_id, src, out, tgt = d
                table.add_data(epoch_num, batch_idx, sample_id, 
                                    wandb.Image(src),
                                    wandb.Image(out), wandb.Image(tgt))

            trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

            if self.first_log:
                self.first_log = False

            self.img_to_log = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        if self.log_wandb:
        
            if not self.first_log:
                api = wandb.Api()
                run_id = self.run_id
                run_project = self.run_project

                artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                    
                if (len(artifs) > 0) and artifs[0].file_count == 1:
                    artifs[0].delete(delete_aliases=True)


class I2IwFiLM_S2_Validation_Log_Reconstructions(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.first_log = True

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)

        return image
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC epoch')
        wandb.define_metric("epoch")
    
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        assert not pl_module.training

        logger = pl_module.logger
        if logger is None:
            return
        
        ####
        tgt = batch['gt_image'] # Target image
        inp = batch['cond_image'] # Conditional image (input)

        batch_size = tgt.shape[0]

        Z, Z_hat = outputs["outputs"]["IPR_S1"], outputs["outputs"]["IPR_S1_hat"],
        D,  reconstructions = outputs["outputs"]["IPR_S2"], outputs["outputs"]["Reconstruction"]
        
        # get all tensors to cpu
        inp, tgt = inp.cpu(), tgt.cpu()
        z, z_hat = Z.cpu(), Z_hat.cpu()
        d, reconstructions = D.cpu(), reconstructions.cpu()

        ####

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")
            cur_out = self.to_dump_format(reconstructions[i], split, batch_idx, bs, dump_type="sample") 
            cur_tgt = self.to_dump_format(tgt[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                    wandb.Image(cur_inp),
                                    wandb.Image(cur_out), wandb.Image(cur_tgt)))


    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.log_wandb:
        
            if not self.first_log:
                api = wandb.Api()
                run_id = wandb.run.id
                run_project = wandb.run.project

                self.run_id = run_id
                self.run_project = run_project
            
                latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-validation_reconstruction:latest')
                if latest_artif.file_count == 1:
                    latest_artif.delete(delete_aliases=True)
                    artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                    if len(artifs) > 0:
                        artifs[0].aliases += ["latest"]
                        artifs[0].save()

            table = wandb.Table(columns=["epoch", "batch", "id", "Input", "Reconstruction", "Target"])
            
            for i, d in enumerate(self.img_to_log):
                epoch_num, batch_idx, sample_id, src, out, tgt = d
                table.add_data(epoch_num, batch_idx, sample_id, 
                                    wandb.Image(src),
                                    wandb.Image(out), wandb.Image(tgt))

            trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

            if self.first_log:
                self.first_log = False

            self.img_to_log = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        if self.log_wandb:
        
            if not self.first_log:
                api = wandb.Api()
                run_id = self.run_id
                run_project = self.run_project

                artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                    
                if (len(artifs) > 0) and artifs[0].file_count == 1:
                    artifs[0].delete(delete_aliases=True)

class VQGAN_Training_Log_Inputs(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32, only_first_epoch=True):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.only_first_epoch = only_first_epoch

        self.first_log = True
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC log_epoch')
        wandb.define_metric("log_epoch")

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)
        # print(f"row shape: {row_png.shape}, dtype: {row_png.dtype}, min: {row_png.min()}, max: {row_png.max()}")

        return image
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None: 
        self.img_to_log = []
    
   
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
         # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        if self.only_first_epoch and trainer.current_epoch > 0:
            return
        
        logger = pl_module.logger
        if logger is None:
            return
        
        ####
        image_key = pl_module.image_key 
        inp = batch[image_key] 
        batch_size = inp.shape[0]
        inp = inp.cpu()

        ####

        split = "train"
        epoch_num = trainer.current_epoch
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                    wandb.Image(cur_inp)))
                
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.only_first_epoch and trainer.current_epoch > 0:
            return
        
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                if not self.first_log:
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project
                
                    latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-train_check_inputs:latest')
                    if latest_artif.file_count == 1:
                        latest_artif.delete(delete_aliases=True)
                        artifs = api.artifacts(name=f'{run_project}/run-{run_id}-train_check_inputs', type_name='run_table')
                        if len(artifs) > 0:
                            artifs[0].aliases += ["latest"]
                            artifs[0].save()

                table = wandb.Table(columns=["epoch", "batch", "id", "Input"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, src = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(src))

                # print("Logging to remote wandb...")
                trainer.logger.experiment.log({"train_check_inputs": table, 'log_epoch':trainer.current_epoch})

                if self.first_log:
                    self.first_log = False
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project

            self.img_to_log = []

class VQGAN_Validation_Log_Reconstructions(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.first_log = True

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)
        # print(f"row shape: {row_png.shape}, dtype: {row_png.dtype}, min: {row_png.min()}, max: {row_png.max()}")

        return image
    
    # def on_training_start(self, trainer, pl_module):        
    def on_train_start(self, trainer, pl_module):  
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC epoch')
        wandb.define_metric("epoch")

        self.run_id = wandb.run.id
        self.run_project = wandb.run.project

    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    # def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        # print(pl_module.training)
        assert not pl_module.training
        
        logger = pl_module.logger
        if logger is None:
            return
        
        ####
        out_dict = outputs["outputs"]
        inp = out_dict["Input"] # Conditional image (input)
        reconstructions = out_dict["Reconstruction"]
        
        batch_size = inp.shape[0]
        
        # get all tensors to cpu
        reconstructions = reconstructions.cpu()
        inp = inp.cpu()

        ####

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")
            cur_out = self.to_dump_format(reconstructions[i], split, batch_idx, bs, dump_type="sample") 

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                        wandb.Image(cur_inp),
                                        wandb.Image(cur_out)))
                
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
        
                if not self.first_log:
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project
                
                    latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-validation_reconstruction:latest')
                    if latest_artif.file_count == 1:
                        latest_artif.delete(delete_aliases=True)
                        artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                        if len(artifs) > 0:
                            artifs[0].aliases += ["latest"]
                            artifs[0].save()

                table = wandb.Table(columns=["epoch", "batch", "id", "Input", "Reconstruction"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, src, out = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(src),
                                        wandb.Image(out))

                trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

                if self.first_log:
                    self.first_log = False

                self.img_to_log = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger = pl_module.logger
        if logger is None:
            return

        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                if not self.first_log:
                    api = wandb.Api()
                    # run_id = wandb.run.id
                    run_id = self.run_id

                    run_project = self.run_project
                    # print(f'run_id: {run_id}, run_project: {run_project}')

                    artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                        
                    if (len(artifs) > 0) and artifs[0].file_count == 1:
                        # print('Last validation-reconstruction artifact was empty, it should be deleted')
                        artifs[0].delete(delete_aliases=True)


    
  


   


class I2IwFiLM_S1_Training_Log_Inputs(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32, only_first_epoch=True):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.only_first_epoch = only_first_epoch

        self.first_log = True
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC log_epoch')
        wandb.define_metric("log_epoch")

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)

        return image
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
         # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        if self.only_first_epoch and trainer.current_epoch > 0:
            return
        
        logger = pl_module.logger
        if logger is None:
            return
        
        ####
        tgt = batch['gt_image'] # Target image
        inp = batch['cond_image'] # Conditional image (input)

        batch_size = tgt.shape[0]

        tgt = tgt.cpu()
        inp = inp.cpu()

        ####

        split = "train"
        epoch_num = trainer.current_epoch
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")
            cur_tgt = self.to_dump_format(tgt[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                        wandb.Image(cur_inp), wandb.Image(cur_tgt)))


    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None: 
        self.img_to_log = []
            
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        
        if self.only_first_epoch and trainer.current_epoch > 0:
            return
        
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(pl_module.logger, WandbLogger):
        
                if not self.first_log:
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project
                
                    latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-train_check_inputs:latest')
                    if latest_artif.file_count == 1:
                        latest_artif.delete(delete_aliases=True)
                        artifs = api.artifacts(name=f'{run_project}/run-{run_id}-train_check_inputs', type_name='run_table')
                        if len(artifs) > 0:
                            artifs[0].aliases += ["latest"]
                            artifs[0].save()

                table = wandb.Table(columns=["epoch", "batch", "id", "Input", "Target"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, src, tgt = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(src), wandb.Image(tgt))

                trainer.logger.experiment.log({"train_check_inputs": table, 'log_epoch':trainer.current_epoch})

                if self.first_log:
                    self.first_log = False
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project

                self.img_to_log = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                if not self.first_log:
                    api = wandb.Api()
                    run_id = self.run_id
                    run_project = self.run_project

                    artifs = api.artifacts(name=f'{run_project}/run-{run_id}-train_check_inputs', type_name='run_table')
                        
                    if (len(artifs) > 0) and artifs[0].file_count == 1:
                        artifs[0].delete(delete_aliases=True)



class I2IwFiLM_S2_Validation_Log_Reconstructions(Callback):
    def __init__(self, output_dir, log_wandb=False, epoch_freq=10, batch_freq=50, max_images=32):
        super().__init__()

        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        
        self.img_to_log = []

        self.first_log = True

    def to_dump_format(self, image, split, batch_idx, bs, dump_type):
        # 1) Output of VQGAN decoder should be in [-1, 1] 
        # print(f"{dump_type} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # 2) rescale to [0, 1]
        image = (image + 1.) / 2.

        # 3) rescale to [0, 255]
        image = image * 255.
        
        # 4) convert to numpy array and to uint8
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.squeeze(image)

        return image
    
    def on_training_start(self, trainer, pl_module):        
        # define a wandb metric to log tables on epochs instead of steps
        print('DEFINING WANDB METRIC epoch')
        wandb.define_metric("epoch")
    
     
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Make sure we are not during sanity check
        if trainer.sanity_checking:
            return

        # Make sure we only log with designated epoch frequency
        if trainer.current_epoch % self.epoch_freq != 0:
            return
        # Make sure we only log with designated batch frequency in the epoch
        if batch_idx % self.batch_freq != 0:
            return
        
        # Save the training status of the model
        assert not pl_module.training
        
        logger = pl_module.logger
        if logger is None:
            return
        
        ####
        out_dict = outputs["outputs"]
        tgt = out_dict["Target"]  # Target image
        inp = out_dict["Input"] # Conditional image (input)
        reconstructions = out_dict["Reconstruction"]
        
        batch_size = tgt.shape[0]
        
        # get all tensors to cpu
        reconstructions = reconstructions.cpu()
        tgt = tgt.cpu()
        inp = inp.cpu()

        ####

        split = "val"
        epoch_num = trainer.current_epoch

        keys = list(outputs.keys())
        bs = batch_size
        for i in range(bs):
            img_idx = batch_idx * bs + i

            sample_id = batch['path'][i]
            sample_id = os.path.basename(sample_id)
            sample_id = os.path.splitext(sample_id)[0]

            cur_inp = self.to_dump_format(inp[i], split, batch_idx, bs, dump_type="condition")
            cur_out = self.to_dump_format(reconstructions[i], split, batch_idx, bs, dump_type="sample") 
            cur_tgt = self.to_dump_format(tgt[i], split, batch_idx, bs, dump_type="target")

            # log to wandb
            if self.log_wandb:
                if isinstance(logger, WandbLogger):
                    self.img_to_log.append((epoch_num, batch_idx, sample_id, 
                                        wandb.Image(cur_inp),
                                        wandb.Image(cur_out), wandb.Image(cur_tgt)))


    def on_validation_epoch_start(self, trainer, pl_module) -> None:     
        self.img_to_log = []
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Make sure we log only with the main process
        if trainer.global_rank != 0:
            return
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                if not self.first_log:
                    api = wandb.Api()
                    run_id = wandb.run.id
                    run_project = wandb.run.project

                    self.run_id = run_id
                    self.run_project = run_project
                
                    latest_artif = api.artifact(name=f'{run_project}/run-{run_id}-validation_reconstruction:latest')
                    if latest_artif.file_count == 1:
                        latest_artif.delete(delete_aliases=True)
                        artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                        if len(artifs) > 0:
                            artifs[0].aliases += ["latest"]
                            artifs[0].save()

                table = wandb.Table(columns=["epoch", "batch", "id", "Input", "Reconstruction", "Target"])
                
                for i, d in enumerate(self.img_to_log):
                    epoch_num, batch_idx, sample_id, src, out, tgt = d
                    table.add_data(epoch_num, batch_idx, sample_id, 
                                        wandb.Image(src),
                                        wandb.Image(out), wandb.Image(tgt))

                trainer.logger.experiment.log({"validation_reconstruction": table, 'epoch':trainer.current_epoch})

                if self.first_log:
                    self.first_log = False

                self.img_to_log = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger = pl_module.logger
        if logger is None:
            return
        
        if self.log_wandb:
            if isinstance(logger, WandbLogger):
                if not self.first_log:
                    api = wandb.Api()
                    run_id = self.run_id
                    run_project = self.run_project

                    artifs = api.artifacts(name=f'{run_project}/run-{run_id}-validation_reconstruction', type_name='run_table')
                        
                    if (len(artifs) > 0) and artifs[0].file_count == 1:
                        artifs[0].delete(delete_aliases=True)
