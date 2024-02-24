import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime

from trainer.common_trainer import LossTracker


class LocalLogger(Logger):

    def __init__(self):
        super(LocalLogger, self).__init__()

    def log_training(
            self,
            epoch: int,
            loss_tracker: LossTracker,
    ):
        train_l1_avg, val_l1_avg, train_perceptual_avg, val_perceptual_avg, train_ssim_avg, val_ssim_avg, train_generator_loss_avg, val_generator_loss_avg, train_discriminator_loss_avg, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()
        print(f"epoch {epoch} train_loss {np.mean(train_loss_avg)} val_loss {np.mean(val_loss_avg)} ssim {np.mean(val_ssim_avg):.5} perceptual {np.mean(val_perceptual_avg):.5}")

    def log_images(
        self, epoch: int, train_images: torch.Tensor, val_images: torch.Tensor
    ):
        pass
