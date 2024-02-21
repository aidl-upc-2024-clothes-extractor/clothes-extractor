import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime


class LocalLogger(Logger):

    def __init__(self):
        super(LocalLogger, self).__init__()

    def log_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        ssim: np.ndarray,
        perceptual: np.ndarray,
    ):
        print(
            f"epoch {epoch} train_loss {np.mean(train_loss_avg):.5} val_loss {np.mean(val_loss_avg):.5} ssim {np.mean(ssim):.5} perceptual {np.mean(perceptual):.5}"
        )

    def log_images(
        self, epoch: int, train_images: torch.Tensor, val_images: torch.Tensor
    ):
        pass
