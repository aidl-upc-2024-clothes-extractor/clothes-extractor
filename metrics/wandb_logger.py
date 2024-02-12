import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime


class WandbLogger(Logger):

    def __init__(
        self,
        wandb_run,
    ):
        self.run = wandb_run


    def log_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        ssim: np.ndarray,
        perceptual: np.ndarray,
        train_generator_loss_avg: np.ndarray,
        eval_generator_loss_avg: np.ndarray,
        train_discriminator_loss_avg: np.ndarray,
    ):
        wandb.log(data={"val_loss": val_loss_avg}, step=epoch)
        wandb.log(data={"train_loss": train_loss_avg}, step=epoch)
        wandb.log(data={"val_ssim": ssim}, step=epoch)
        wandb.log(data={"val_perceptual": perceptual}, step=epoch)
        wandb.log(data={"train_generator_loss": train_generator_loss_avg}, step=epoch)
        wandb.log(data={"eval_generator_loss": eval_generator_loss_avg}, step=epoch)
        wandb.log(data={"train_discriminator_loss": train_discriminator_loss_avg}, step=epoch)

