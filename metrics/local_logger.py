import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime

from trainer.trainer import LossTracker


class LocalLogger(Logger):

    def __init__(self):
        pass

    def log_training(
            self,
            epoch: int,
            loss_tracker: LossTracker,
    ):
        train_l1_avg, val_l1_avg, train_perceptual_avg, val_perceptual_avg, train_ssim_avg, val_ssim_avg, train_generator_loss_avg, val_generator_loss_avg, train_discriminator_loss_avg, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()
        print(f"epoch {epoch} train_loss {np.mean(train_loss_avg)} val_loss {np.mean(val_loss_avg)}")

