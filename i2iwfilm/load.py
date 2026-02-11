import collections
from typing import Any, Dict, Mapping, Optional, Tuple
from hydra.utils import instantiate
from hydra import initialize_config_module as init_hydra, compose, initialize_config_dir
from hydra import initialize
from hydra.core.utils import configure_log
from mlflow.tracking.client import MlflowClient
import omegaconf
from omegaconf import ListConfig, OmegaConf, DictConfig
import mlflow
import pytorch_lightning as pl
from tempfile import TemporaryDirectory
from importlib import import_module
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf ,open_dict

from pathlib import Path
import torch

from . import module

from typing import List, Union
import numpy as np

def pick_gpu(number: Union[int, List]):
    from subprocess import check_output

    if isinstance(number, List):
        return number

    gpu_stats = [
        int(x.split(", ")[0]) / int(x.split(", ")[1])
        for x in check_output(
            [
                "nvidia-smi",
                "--format=csv,nounits,noheader",
                "--query-gpu=memory.used,memory.total",
            ]
        )
        .decode()
        .strip()
        .split("\n")
    ]
    if len(gpu_stats) < number:
        raise ValueError(f"Not enough GPU's for the requested {number}.")

    return [int(x) for x in np.argsort(gpu_stats)[:number]]


def load_from_dir_without_hyperparam_in_statedict(run_path, model_path=None, load_trainer=False, override=None) -> Tuple:
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    with initialize_config_dir(config_dir=str(config_path.parents[0])):
        cfg = compose(
            config_name="config", overrides=override, return_hydra_config=True
        )
    
    model: LightningModule = instantiate(cfg.module, _recursive_=False)
    if model_path is None:
        model_path = run_path / "checkpoints/last.ckpt"
    else:
        model_path = run_path / 'checkpoints' / model_path
    model.load_state_dict(torch.load(model_path)['state_dict'])

    
    print(f'loded model from {model_path}')

    print('loading datamodule')
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()

    print('loading trainer')

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        if isinstance(cfg.logger, ListConfig) and len(cfg.logger) == 0:
            logger = None
        else:
            tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
            logger = instantiate(tmp_logger)

        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.gpus, int):
            cfg.trainer.gpus = pick_gpu(cfg.trainer.gpus)

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, model, datamodule, trainer



def load_from_dir_without_hyperparam_in_statedict_new_pl(run_path, model_path=None, load_trainer=False, override=None) -> Tuple:
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    # Due to new PyTorch Lightning serialization safety mechanism, we need to add safe globals
    torch.serialization.add_safe_globals([Any])   
    torch.serialization.add_safe_globals([dict])
    torch.serialization.add_safe_globals([list])
    torch.serialization.add_safe_globals([ListConfig])
    torch.serialization.add_safe_globals([DictConfig])
    torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata]) 
    torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
    torch.serialization.add_safe_globals([omegaconf.base.Metadata])
    torch.serialization.add_safe_globals([collections.defaultdict])
    torch.serialization.add_safe_globals([int])

    with initialize_config_dir(config_dir=str(config_path.parents[0])):
        cfg = compose(
            config_name="config", overrides=override, return_hydra_config=True
        )
    
    model: LightningModule = instantiate(cfg.module, _recursive_=False)
    if model_path is None:
        model_path = run_path / "checkpoints/last.ckpt"
    else:
        model_path = run_path / 'checkpoints' / model_path
    model.load_state_dict(torch.load(model_path)['state_dict'])

    print(f'loded model from {model_path}')

    print('loading datamodule')
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()

    print('loading trainer')

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        if isinstance(cfg.logger, ListConfig) and len(cfg.logger) == 0:
            logger = False
        else:
            tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
            logger = instantiate(tmp_logger)

        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.devices, int):
            cfg.trainer.devices = pick_gpu(cfg.trainer.devices)

        if "resume_from_checkpoint" in cfg.trainer:
            with open_dict(cfg.trainer):
                # remove the resume_from_checkpoint key from the trainer config
                cfg.trainer.pop("resume_from_checkpoint")

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, model, datamodule, trainer





def load_from_dir(run_path, model_path=None, load_trainer=False, override=None) -> Tuple:
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    with initialize_config_dir(config_dir=str(config_path.parents[0])):
        cfg = compose(
            config_name="config", overrides=override, return_hydra_config=True
        )

    print(cfg)
    

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = run_path / "checkpoints/last.ckpt"
    else:
        model_path = run_path / 'checkpoints' / model_path
    print(module_class)
    model = module_class.load_from_checkpoint(model_path)
    
    print(f'loded model from {model_path}')

    print('loading datamodule')
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()

    print('loading trainer')

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
        logger = instantiate(tmp_logger)
        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.gpus, int):
            cfg.trainer.gpus = pick_gpu(cfg.trainer.gpus)

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, model, datamodule, trainer



def load_from_dir_new_pl(run_path, model_path=None, load_trainer=False, override=None) -> Tuple:
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    with initialize_config_dir(config_dir=str(config_path.parents[0])):
        cfg = compose(
            config_name="config", overrides=override, return_hydra_config=True
        )

    print(cfg)
    

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = run_path / "checkpoints/last.ckpt"
    else:
        model_path = run_path / 'checkpoints' / model_path
    print(module_class)
    model = module_class.load_from_checkpoint(model_path)
    
    print(f'loded model from {model_path}')

    print('loading datamodule')
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()

    print('loading trainer')

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
        logger = instantiate(tmp_logger)
        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.devices, int):
            with open_dict(cfg.trainer):
                cfg.trainer.devices = pick_gpu(cfg.trainer.devices)


        if "resume_from_checkpoint" in cfg.trainer:
            # remove the resume_from_checkpoint key from the trainer config
            cfg.trainer.pop("resume_from_checkpoint")

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, model, datamodule, trainer




def load_from_overrides_and_modelpath (overrides=[], model_path=None , load_trainer=False) -> Tuple:
    with init_hydra(config_module="sunscc.conf"):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )
        print(type(cfg))

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = Path('.') / "models/last.ckpt"
    model = module_class.load_from_checkpoint(model_path)


    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        logger = instantiate(cfg.logger)
        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.gpus, int):
            cfg.trainer.gpus = pick_gpu(cfg.trainer.gpus)

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    
    return cfg, model, datamodule, trainer