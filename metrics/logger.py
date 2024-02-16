import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

class Logger:

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
        raise NotImplementedError

    def log_images(self, epoch:int, train_images: torch.Tensor, val_images: torch.Tensor):
        raise NotImplementedError