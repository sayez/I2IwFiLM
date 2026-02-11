import os 
import math
import shutil
from typing import Any
from bisect import bisect_right

import collections
import omegaconf
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure,\
                                            structural_similarity_index_measure
import pytorch_lightning as pl

class CustomSequentialLR(SequentialLR):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, maintain_lr=False):
        super().__init__(optimizer, schedulers, milestones, last_epoch)
        self.maintain_lr = maintain_lr
        self.last_scheduler_index = 0

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)

        scheduler = self._schedulers[idx]

        if (idx != self.last_scheduler_index) and self.maintain_lr: 
            # make sure that we maintain LR continuity with last scheduler
            scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

def compute_L1_loss(x, x_hat):
    return F.l1_loss(x_hat, x)

def compute_L2_loss(x, x_hat):
    return F.mse_loss(x_hat, x)

def compute_reconstruction_loss(x, x_hat):
    return compute_L1_loss(x, x_hat)

def compute_diffusion_loss(x, x_hat):
    return compute_L1_loss(x, x_hat)

def compute_ms_ssim_loss(x, x_hat):
    data_range = (-1, 1)
    ms_ssim = multiscale_structural_similarity_index_measure(x, x_hat, data_range=data_range, normalize='relu')
    return 1 - ms_ssim

def compute_patched_ms_ssim_loss(x, x_hat, patch_size=16):
    # compute the multiscale ssim loss for each patch
    patched_x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

def compute_FFT_L1_loss(x, x_hat):
    x_fft = torch.fft.fft2(x)
    x_hat_fft = torch.fft.fft2(x_hat)
    # compute the logabs of the fft
    x_fft = torch.log(torch.abs(x_fft) + 1e-6)
    x_hat_fft = torch.log(torch.abs(x_hat_fft) + 1e-6)

    return compute_L1_loss(x_fft, x_hat_fft)

class Base_I2IwFiLM_S1_Module(pl.LightningModule):
    def __init__(self, 
                    in_channels: int,
                    in_size: int,
                    model_config: Any,
                    loss_weights= [.5,.5],
                    optimizer = None,
                    scheduler = None,
                    scheduler_interval = "epoch",
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.in_channels = in_channels
        self.in_size = in_size

        # loss weights order: [reconstruction (L1), FFT_L1]
        self.loss_weights = loss_weights
        if len(loss_weights) == 3:
            self.l_l1, self.l_fft_l1, self.l_ms_ssim = loss_weights
            print(f"Loss weights: {self.l_l1} (L1), {self.l_fft_l1} (FFT), {self.l_ms_ssim} (MS-SSIM)")
            self.l_l2 = None
        elif len(loss_weights) == 4:
            self.l_l1, self.l_fft_l1, self.l_ms_ssim, self.l_l2 = loss_weights
            print(f"Loss weights: {self.l_l1} (L1), {self.l_fft_l1} (FFT), {self.l_ms_ssim} (MS-SSIM), {self.l_l2} (L2)")
            
        self.model = instantiate(model_config)

        print("the number of MODEL parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def training_step(self, batch, batch_idx):
        tgt = batch['gt_image'] # Target image
        inp = batch['cond_image'] # Conditional image (input)

        tgt_name = batch['path']
        inp_name = batch['cond_path']

        batch_size = tgt.shape[0]

        output, output_IPR = self.model(inp, tgt)

        # compute reconstruction loss
        loss_dict = {"reconstruction": compute_reconstruction_loss(tgt, output),
                     "FFT_L1": compute_FFT_L1_loss(tgt, output),
                     "ms_ssim": compute_ms_ssim_loss(tgt, output)}
        if self.l_l2 is not None:
            loss_dict["L2"] = compute_L2_loss(tgt, output)
        
        # check that loss_dict[reconstruction] is a tensor with no nan values
        if torch.isnan(loss_dict["reconstruction"]).any():
            # get the indices of the NaN values
            idx = torch.isnan(loss_dict["reconstruction"])
            # print the indices of the NaN values
            print(loss_dict["reconstruction"])

            print(f'Indices of NaN values in the reconstruction loss: {idx}')

            print(inp_name)
            print(tgt_name)

            raise ValueError('Reconstruction loss contains NaN values')

        loss_means = { k: l.mean() for k, l in loss_dict.items()}

        # log loss
        for k, loss in loss_means.items():
            self.log(f'train/{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f'train_{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        total_loss = self.l_l1 * loss_dict["reconstruction"] + self.l_fft_l1 *loss_dict["FFT_L1"] + self.l_ms_ssim * loss_dict["ms_ssim"]
        if self.l_l2 is not None:
            total_loss += self.l_l2 * loss_dict["L2"]
            
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)
        
        outputs_dict = {
            "Input": inp,
            'Target': tgt,
            "Reconstruction": output,
            "IPR": output_IPR
        }

        return dict(loss = total_loss, outputs=outputs_dict)
    
    def validation_step(self, batch, batch_idx):
        tgt = batch['gt_image']
        inp = batch['cond_image']
        
        tgt_name = batch['path']
        inp_name = batch['cond_path']

        batch_size = tgt.shape[0]

        output, output_IPR = self.model(inp, tgt)

        # compute reconstruction loss
        loss_dict = {"reconstruction": compute_reconstruction_loss(tgt, output),
                     "FFT_L1": compute_FFT_L1_loss(tgt, output),
                     "ms_ssim": compute_ms_ssim_loss(tgt, output)}
        if self.l_l2 is not None:
            loss_dict["L2"] = compute_L2_loss(tgt, output)

        loss_means = { k: l.mean() for k, l in loss_dict.items()}
        # log loss
        for k, loss in loss_means.items():
            self.log(f'val/{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f'val_{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        total_loss = self.l_l1 * loss_dict["reconstruction"] + self.l_fft_l1 *loss_dict["FFT_L1"] + self.l_ms_ssim * loss_dict["ms_ssim"]
        if self.l_l2 is not None:
            total_loss += self.l_l2 * loss_dict["L2"]
        
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)
        
        outputs_dict = {
            "Input": inp,
            'Target': tgt,
            "Reconstruction": output,
            "IPR": output_IPR
        }

        return dict(loss = total_loss, outputs=outputs_dict)

    
    def test_step(self, batch, batch_idx):
        x = batch['gt_image']
        x_cond = batch['cond_image']
        x_name = batch['path']
        x_cond_name = batch['cond_path']

        pass


    def configure_optimizers(self):
        if self.scheduler is not None:
            optimizer = {
                    "lr": self.scheduler.init_lr,
                    **self.optimizer,
                }
            optimizer = instantiate(optimizer, params=self.model.parameters())

            if (self.scheduler._target_ == 'torch.optim.lr_scheduler.SequentialLR') or \
                (self.scheduler._target_ == 'i2iwfilm.module.CustomSequentialLR'):
                scheds = []
                milestones = self.scheduler.milestones
                batches_per_epoch = math.ceil(len(self.trainer.datamodule.train_ds) // self.trainer.datamodule.batch_size)
                if self.scheduler_interval == "step":
                    milestones = [i * batches_per_epoch for i in milestones]

                for i, (s, ms) in enumerate(zip(self.scheduler.schedulers, milestones)):
                    if s._target_ == 'torch.optim.lr_scheduler.MultiStepLR':
                        if self.scheduler_interval == "step":
                            s.milestones = [i * batches_per_epoch for i in s.milestones]
                    if s._target_ == 'torch.optim.lr_scheduler.CosineAnnealingLR':
                        s.T_max = s.T_max - milestones[i - 1] if i > 0 else s.T_max
                    if s._target_ == 'torch.optim.lr_scheduler.LinearLR':
                        if self.scheduler_interval == "step":
                            s.total_iters = s.total_iters * batches_per_epoch
                    print(s)
                    scheds.append(instantiate(s, optimizer=optimizer))

                assert len(scheds) == len(self.scheduler.milestones)

                final_sched = None
                if self.scheduler._target_ == 'i2iwfilm.module.CustomSequentialLR':
                    final_sched = CustomSequentialLR(optimizer, schedulers=scheds, 
                                                     milestones=milestones[:-1],maintain_lr=self.scheduler.maintain_lr )
                elif self.scheduler._target_ == 'torch.optim.lr_scheduler.SequentialLR':
                    final_sched = SequentialLR(optimizer, schedulers=scheds, 
                                                     milestones=milestones[:-1])

                return dict(
                    optimizer=optimizer,
                    lr_scheduler={
                        "scheduler": final_sched,
                        "interval": self.scheduler_interval,
                        "frequency": 1,
                    },
                )
            else:
                scheduler = instantiate(self.scheduler, optimizer=optimizer)
                return dict(
                    optimizer=optimizer,
                    lr_scheduler={
                        "scheduler": scheduler,
                        "interval": self.scheduler.interval,
                        "frequency": 1,
                        "reduce_on_plateau": True,
                        "monitor": "val/loss",
                    },
                )
        
        else:
            optimizer = instantiate(self.optimizer, params=self.model.parameters())  
            return [optimizer]
        


class Base_I2IwFiLM_S2_Module(pl.LightningModule):
    # this is the updated version of the I2IwFiLM S2 module (handling several losses)
    # I2IwFiLM uses a MLP instead of a diffusion model for the second stage, as it is done in DiffI2I.
    def __init__(self, 
                    in_channels: int,
                    in_size: int,
                    model_config: Any,
                    diff_former_loss_weights= [.33,.33,.33],
                    diffusion_loss_weights= [1.],
                    optimizer = None,
                    scheduler = None,
                    scheduler_interval = "epoch",
                    load_S1_weights = True, # whether to load/dump the initial weights of the S1 model in the current run directory, 
                                              # typically True for training, False for testing and inference.
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.in_channels = in_channels
        self.in_size = in_size


        # loss weights order: [reconstruction (L1), FFT_L1]
        self.diff_former_loss_weights = diff_former_loss_weights
        if len(diff_former_loss_weights) == 3:
            self.diff_former_l_l1, self.diff_former_l_fft_l1, self.diff_former_l_ms_ssim = diff_former_loss_weights
            self.diff_former_l_l2 = None
        elif len(diff_former_loss_weights) == 4:
            self.diff_former_l_l1, self.diff_former_l_fft_l1, self.diff_former_l_ms_ssim, self.diff_former_l_l2 = diff_former_loss_weights


        # loss weights order: [reconstruction (L1), FFT_L1]
        self.diffusion_loss_weights = diffusion_loss_weights
        self.diffusion_l_l1 = diff_former_loss_weights[0]

        # Model S1 is loaded from the checkpoint of the previous run
        s1_run_dir = model_config.s1_run_dir

        # check that s1_run_dir is an absolute path
        if not os.path.isabs(s1_run_dir):
            # This must be an absolute path, as the current working directory is set 
            # to be the current run directory when using hydra
            raise ValueError(f's1_run_dir must be an absolute path, got {s1_run_dir}')
        # another possibility is to get the absolute path by joining the received path with the
        # I2IwFiLM root directory (in environment variable I2IWFILM_PATH)

        # a) from run dir, get the model config located in .hydra/config.yaml
        s1_model_config = OmegaConf.load(os.path.join(s1_run_dir, '.hydra/config.yaml')).model
        print(f'S1 model config loaded from {s1_model_config}')
        s1_model_path = os.path.join(s1_run_dir, model_config.s1_model_path)
        

        self.model_S1 = instantiate(s1_model_config)
        
        print(f'Copying the checkpoint from S1 run to current S2 run')
        src_path = s1_model_path if not os.path.islink(s1_model_path) else os.readlink(s1_model_path)
        destination_dir = os.path.join(".", 'S2_checkpoints')
        destination_path = os.path.join(destination_dir, "S1_last.ckpt")
        print(f"src_path: {src_path}\ndestination_path: {destination_path}")

        if load_S1_weights:
            # copy the weights of the I2IwFiLMFormer from S1 directory to current directory
            os.makedirs(destination_dir, exist_ok=True)    
            # Copy the file from source_path to destination_path
            shutil.copy(src_path, destination_path)

            # ATTENTION, with versions of torch>2.6 , setting weights_only=False is dangerous 
            # as it may lead to arbitrary code execution, use only if you trust the source of the checkpoint
            # In our case, we trust the source as we are loading our own checkpoints
            # See torch docs for more details.
            tmp = torch.load(destination_path, weights_only=False)['state_dict']

            # at this point, tmp is a dict with keys that are prefixed with 'model.'
            # change the keys to remove the prefix 'model.'
            tmp = {k.replace('model.', ''): v for k, v in tmp.items()}
            
            self.model_S1.load_state_dict(tmp)
            self.model_S1.eval()

        s2_model_config = model_config.model_s2
        # remove the '_target_' key from the S1 model config
        s1_model_config.pop('_target_', None)
        s2_model_config = {**s2_model_config, **s1_model_config}
 
        self.model_S2 = instantiate(s2_model_config)
    
    def on_train_start(self) -> None:
        if self.trainer.current_epoch > 0:
            print("This is a resumed training run (epoch > 0 ), DO NOT RELOAD THE WEIGHTS OF THE S1 MODEL")
        else:
            # copy the weights of I2IwFiLMFormer from S1 to S2
            print('Copying the weights of the I2IwFiLMFormer from S1 to S2')
            try:
                self.model_S2.STM.load_state_dict(self.model_S1.STM.state_dict())
            except AttributeError:
                print('The model does not have the STM module, maybe was named G')
                self.model_S2.STM.load_state_dict(self.model_S1.G.state_dict())

        # Make sure that the weights of the S1 model are not updated but 
        # the weights of the S2 model are updated
        print('Freezing the weights of the S1 model (GVP_P + I2IwFiLMFormer_S1)')
        for param in self.model_S1.parameters():
            param.requires_grad = False
        self.model_S1.eval()
        
        print('Updating the weights of the S2 model (GVP_W + I2IwFiLMFormer_S2)')
        for param in self.model_S2.parameters():
            param.requires_grad = True

    def training_step(self, batch, batch_idx):
        tgt = batch['gt_image']
        inp = batch['cond_image']
        
        tgt_name = batch['path']
        inp_name = batch['cond_path']

        batch_size = tgt.shape[0]

        # 1) Get the IPR from the GVP_P (that concatenates the input and the gt)
        _, S1_IPR = self.model_S1(inp, tgt) # Z
        Ep = S1_IPR[0]

        # Z is ready for forward diffusion process, should be 4*C' (4*64=256)
        # This is done by calling the forward method of model_S2
        # 2)  Get the reconstruction from the model_S2
        #    a) compute D, the IPR from the GVP_W (takes only the input)
        #    b) Forward diffusion process + Backward diffusion process
        #       Z -|Forward|-> Z_T  // Z_T -|Backward|-> Z_0_hat
        #    c) Get the reconstruction from the I2IwFiLMFormer_S2
        #       (input, Z_0_hat) -|I2IwFiLMFormer_S2|-> output

        output, Ep_hat, Ew = self.model_S2(inp)

        # compute L_all loss s.t.  L_all = L_task +  L_diff 
        # L_diff = Eq. 13 in the DiffI2I paper (L1 loss between Z and Z_hat)
        diff_loss_dict = {"diffusion": compute_diffusion_loss(Ep, Ep_hat)}
        L_diff = self.diffusion_l_l1 * diff_loss_dict["diffusion"]


        # L_task = L_recon  (L1 loss between gt and reconstructed image)
        rec_loss_dict = {"reconstruction": compute_reconstruction_loss(tgt, output),
                        "FFT_L1": compute_FFT_L1_loss(tgt, output),
                        "ms_ssim": compute_ms_ssim_loss(tgt, output)}
        if self.diff_former_l_l2 is not None:
            rec_loss_dict["L2"] = compute_L2_loss(tgt, output)
        
        L_task = (self.diff_former_l_l1 * rec_loss_dict["reconstruction"] + 
                  self.diff_former_l_fft_l1 * rec_loss_dict["FFT_L1"] + 
                  self.diff_former_l_ms_ssim * rec_loss_dict["ms_ssim"])
        if self.diff_former_l_l2 is not None:
            L_task += self.diff_former_l_l2 * rec_loss_dict["L2"]

        loss_dict = {}
        for k, l in rec_loss_dict.items():
            loss_dict[k] = l
        loss_dict['L_task'] = L_task
        loss_dict["L_diff"] = L_diff

        # log loss
        loss_means = { k: l.mean() for k, l in loss_dict.items()}
        for k, loss in loss_means.items():
            self.log(f'train/{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f'train_{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        loss = L_task + L_diff

        self.log('train/S2_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train_S2_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        outputs_dict = {
            "Input": inp,
            'Target': tgt,
            "Reconstruction": output,
            "IPR_S1": Ep,
            "IPR_S1_hat": Ep_hat,
            "IPR_S2": Ew,
        }
        return dict(loss = loss, outputs=outputs_dict)
    
    def validation_step(self, batch, batch_idx):
        # print('VALIDATION STEP')
        tgt = batch['gt_image']
        inp = batch['cond_image']
        
        tgt_name = batch['path']
        inp_name = batch['cond_path']

        batch_size = inp.shape[0]

        # 1) Get the IPR from the GVP_P (that concatenates the input and the gt)
        _, S1_IPR = self.model_S1(inp, tgt) # Ep
        Ep = S1_IPR[0]

        # Ep is ready for forward diffusion process, should be 4*C' (4*64=256)
        # This is done by calling the forward method of model_S2
        # 2)  Get the reconstruction from the model_S2
        #    a) compute D, the IPR from the GVP_W (takes only the input)
        #    b) Forward diffusion process + Backward diffusion process
        #       Z -|Forward|-> Z_T  // Z_T -|Backward|-> Z_0_hat
        #    c) Get the reconstruction from the I2IwFiLMFormer_S2
        #       (input, Z_0_hat) -|I2IwFiLMFormer_S2|-> output

        output, Ep_hat, Ew = self.model_S2(inp) # reconstructed IPR from backward diffusion process

        # compute L_all loss s.t.  L_all = L_task +  L_diff 
        L_diff = compute_diffusion_loss(Ep, Ep_hat)

        # L_task = L_recon  (L1 loss between gt and reconstructed image)
        rec_loss_dict = {"reconstruction": compute_reconstruction_loss(tgt, output),
                        "FFT_L1": compute_FFT_L1_loss(tgt, output),
                        "ms_ssim": compute_ms_ssim_loss(tgt, output)}
        if self.diff_former_l_l2 is not None:
            rec_loss_dict["L2"] = compute_L2_loss(tgt, output)
        
        L_task = (self.diff_former_l_l1 * rec_loss_dict["reconstruction"] +
                  self.diff_former_l_fft_l1 * rec_loss_dict["FFT_L1"] +
                  self.diff_former_l_ms_ssim * rec_loss_dict["ms_ssim"])
        if self.diff_former_l_l2 is not None:
            L_task += self.diff_former_l_l2 * rec_loss_dict["L2"]

        loss_dict = {}
        for k, l in rec_loss_dict.items():
            loss_dict[k] = l
        loss_dict['L_task'] = L_task
        loss_dict["L_diff"] = L_diff

        # log loss
        loss_means = { k: l.mean() for k, l in loss_dict.items()}
        for k, loss in loss_means.items():
            self.log(f'val/{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f'val_{k}_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        loss = L_task + L_diff

        self.log('val/S2_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_S2_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False, batch_size=batch_size)

        outputs_dict = {
            "Input": inp,
            'Target': tgt,
            "Reconstruction": output,
            "IPR_S1": Ep,
            "IPR_S1_hat": Ep_hat,
            "IPR_S2": Ew,
        }

        return dict(loss = loss, outputs=outputs_dict)

    def predict(self, batch):
        inp = batch['cond_image']
        inp_name = batch['cond_path']
        
        output, Ep_hat, Ew = self.model_S2(inp) # reconstructed IPR from backward diffusion process
        
        outputs_dict = {
            "Input": inp,
            "Reconstruction": output,
            "Image_name": inp_name,
        }

        return dict(outputs=outputs_dict)
    
    def test_step(self, batch, batch_idx):
        tgt = batch['gt_image']
        inp = batch['cond_image']
        
        tgt_name = batch['path']
        inp_name = batch['cond_path']

        output, Ep_hat, Ew = self.model_S2(inp) # reconstructed IPR from backward diffusion process
        
        outputs_dict = {
            "Input": inp,
            'Target': tgt,
            "Reconstruction": output,
            "IPR_S1": None,
            "IPR_S1_hat": Ep_hat,
            "IPR_S2": Ew,
        }

        return dict(outputs=outputs_dict)


    def configure_optimizers(self):
        import itertools  

        if self.scheduler is not None:
            optimizer = {
                    "lr": self.scheduler.init_lr,
                    **self.optimizer,
                }
            optimizer = instantiate(optimizer, params=self.model_S2.parameters())

            if (self.scheduler._target_ == 'torch.optim.lr_scheduler.SequentialLR') or \
                (self.scheduler._target_ == 'i2iwfilm.module.CustomSequentialLR'):
                scheds = []
                milestones = self.scheduler.milestones
                
                batches_per_epoch = math.ceil(len(self.trainer.datamodule.train_ds) // self.trainer.datamodule.batch_size)
                
                if self.scheduler_interval == "step":
                    milestones = [i * batches_per_epoch for i in milestones]
                
                for i, (s, ms) in enumerate(zip(self.scheduler.schedulers, milestones)):
                    if s._target_ == 'torch.optim.lr_scheduler.MultiStepLR':
                        if self.scheduler_interval == "step":
                            s.milestones = [i * batches_per_epoch for i in s.milestones]
                    if s._target_ == 'torch.optim.lr_scheduler.CosineAnnealingLR':
                        s.T_max = s.T_max - milestones[i - 1] if i > 0 else s.T_max
                    if s._target_ == 'torch.optim.lr_scheduler.LinearLR':
                        if self.scheduler_interval == "step":
                            s.total_iters = s.total_iters * batches_per_epoch
                    print(s)
                    scheds.append(instantiate(s, optimizer=optimizer))

                assert len(scheds) == len(self.scheduler.milestones)

                final_sched = None
                
                if self.scheduler._target_ == 'i2iwfilm.module.CustomSequentialLR':
                    final_sched = CustomSequentialLR(optimizer, schedulers=scheds, 
                                                     milestones=milestones[:-1],maintain_lr=self.scheduler.maintain_lr )
                elif self.scheduler._target_ == 'torch.optim.lr_scheduler.SequentialLR':
                    final_sched = SequentialLR(optimizer, schedulers=scheds, 
                                                     milestones=milestones[:-1])

                return dict(
                    optimizer=optimizer,
                    lr_scheduler={
                        "scheduler": final_sched,
                        "interval": self.scheduler_interval,
                        "frequency": 1,
                    },
                )
            else:
                scheduler = instantiate(self.scheduler, optimizer=optimizer)
                return dict(
                    optimizer=optimizer,
                    lr_scheduler={
                        "scheduler": scheduler,
                        "interval": self.scheduler.interval,
                        "frequency": 1,
                        "reduce_on_plateau": True,
                        "monitor": "val/loss",
                    },
                )
        
        else:
            optimizer = instantiate(self.optimizer, params=self.model_S2.parameters())  
            return [optimizer]
