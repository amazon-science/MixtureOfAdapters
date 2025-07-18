# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
import uuid

import torch
import logging

from lightning.pytorch import Trainer

from mixture_of_adapters.products_datamodule import TripletDataModule
from mixture_of_adapters.products_module import TripletModule

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate


def create_logdirectory(task_type):
    """
    Create a directory for logs.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join("logs", task_type, f"triplet_model_{timestamp}_{str(uuid.uuid4())[:6]}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

@hydra.main(config_path="config", config_name="train_products", version_base="1.3")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Set the precision for float32 matrix multiplication to use Tensor cores
    torch.set_float32_matmul_precision('medium')

    if 'logdir' not in cfg or cfg['logdir'] is None:
        task_type = cfg['task_type']
        logdir = create_logdirectory(task_type)
        cfg['logdir'] = logdir
    else:
        logdir = cfg['logdir']

    if 'dump_stderr' in cfg and cfg['dump_stderr']:
        stderr_path = os.path.join(logdir, "log.err")
        stderr_handler = logging.FileHandler(stderr_path, mode='w')
        stderr_handler.setLevel(logging.ERROR)
        logger.addHandler(stderr_handler)

    if 'dump_stdout' in cfg and cfg['dump_stdout']:
        # Set up stdout logging
        stdout_path = os.path.join(logdir, "log.out")
        stdout_handler = logging.FileHandler(stdout_path, mode='w')
        stdout_handler.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)
        logging.getLogger("lightning.pytorch").handlers.clear()
        logging.getLogger("lightning.pytorch").addHandler(stdout_handler)

        # disable progress bar and remove it from callbacks
        cfg.trainer.enable_progress_bar = False
        callbacks = cfg.trainer.callbacks
        callbacks_kept = []
        for callback in callbacks:
            if hasattr(callback, '_target_') and 'ProgressBar' in callback._target_:
                continue
            callbacks_kept.append(callback)
        cfg.trainer.callbacks = callbacks_kept
    else:
        logging.basicConfig()

    logger.info(f"Log directory created at: {logdir}")

    # Initialize the data module
    datamodule: TripletDataModule = instantiate(cfg.datamodule)

    # Initialize the model
    model: TripletModule = instantiate(cfg.module, _recursive_=False)    

    # Initialize the trainer
    trainer: Trainer = instantiate(cfg.trainer)

    # Save the model configuration
    model_config_path = os.path.join(logdir, "config.yaml")
    OmegaConf.save(cfg, model_config_path)

    # Train the model
    trainer.fit(model, datamodule)

    # Retrieve the best validation loss
    best_id_val_loss = trainer.checkpoint_callbacks[0].best_model_score
    best_ood_val_loss = trainer.checkpoint_callbacks[1].best_model_score

    return best_id_val_loss, best_ood_val_loss

if __name__ == "__main__":
     # Set CUDA visible devices
    # parser = argparse.ArgumentParser(description="Train the triplet model.")
    # parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU devices to use.")
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    torch.autograd.set_detect_anomaly(True)

    main()
