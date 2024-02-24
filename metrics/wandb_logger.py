import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime

from trainer.common_trainer import LossTracker

class WandbLogger(Logger):

    def __init__(
        self,
        wandb_run,
    ):
        super(WandbLogger, self).__init__()
        self.run = wandb_run

    def log_training(
        self,
        epoch: int,
        loss_tracker: LossTracker,
    ):

        train_l1_avg, val_l1_avg, train_perceptual_avg, val_perceptual_avg, train_ssim_avg, val_ssim_avg, train_generator_loss_avg, val_generator_loss_avg, train_discriminator_loss_avg, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()

        wandb.log({"val_loss": val_loss_avg}, step=epoch)
        wandb.log({"train_loss": train_loss_avg}, step=epoch)
        wandb.log({"val_l1": val_l1_avg}, step=epoch)
        wandb.log({"train_l1": train_l1_avg}, step=epoch)
        wandb.log({"val_ssim": val_ssim_avg}, step=epoch)
        wandb.log({"train_ssim": train_ssim_avg}, step=epoch)
        wandb.log({"val_perceptual": val_perceptual_avg}, step=epoch)
        wandb.log({"train_perceptual": train_perceptual_avg}, step=epoch)
        wandb.log({"train_generator_loss": train_generator_loss_avg}, step=epoch)
        wandb.log({"val_generator_loss": val_generator_loss_avg}, step=epoch)
        wandb.log({"train_discriminator_loss": train_discriminator_loss_avg}, step=epoch)

    def log_images(self, epoch:int, train_images: torch.Tensor, val_images: torch.Tensor, train_target: torch.Tensor, val_target: torch.Tensor):
        wandb.log({"train_images": [wandb.Image(img) for img in train_images]}, step=epoch)
        wandb.log({"val_images": [wandb.Image(img) for img in val_images]}, step=epoch)
        wandb.log({"train_target": [wandb.Image(img) for img in train_target]}, step=epoch)
        wandb.log({"val_target": [wandb.Image(img) for img in val_target]}, step=epoch)
