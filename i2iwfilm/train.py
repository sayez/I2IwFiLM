import os
from pathlib import Path
from typing import Mapping
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import call, instantiate
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import torch


@hydra.main(config_path="config", config_name="base_config", version_base='1.2' )
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('medium')

    seed_everything(cfg.seed, workers=True)
    # check if we are resuming from a checkpoint
    latest_checkpoint = Path(os.getcwd()) / "checkpoints/last.ckpt"
    latest_checkpoint = latest_checkpoint if latest_checkpoint.exists() else None

    # check if we are asked to resume from a specific checkpoint
    if cfg.trainer.resume_from_checkpoint is not None:
        print(cfg.trainer.resume_from_checkpoint)
        latest_checkpoint = Path(os.getcwd()) /  ("checkpoints/"+cfg.trainer.resume_from_checkpoint)
        latest_checkpoint = latest_checkpoint if latest_checkpoint.exists() else None
        print(latest_checkpoint)

    if latest_checkpoint is not None:
        print(f"Found checkpoint at {latest_checkpoint}")
    else:
        print("Starting from scratch")

    datamodule = instantiate(cfg.dataset, _recursive_=False)

    print(cfg.module)

    module = instantiate(cfg.module, _recursive_=False)

    callbacks = []
    if isinstance(cfg.callbacks, Mapping):
        cfg.callbacks = [cb for cb in cfg.callbacks.values()]
    for callback in cfg.callbacks:
        callback = instantiate(callback, _recursive_=False)
        callback.cfg = cfg  # FIXME : ugly hack
        callbacks.append(callback)

    with open_dict(cfg):
        cfg.trainer.pop('resume_from_checkpoint', None)

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=cfg.logger,
        default_root_dir=".",
        callbacks=callbacks,
        # resume_from_checkpoint=latest_checkpoint,
        _recursive_=True,
        _convert_="all",
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=latest_checkpoint)

    return {key: value.item() for key, value in trainer.callback_metrics.items()}


if __name__ == "__main__":
    main()
