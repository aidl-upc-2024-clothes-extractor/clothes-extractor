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
    ):
        raise NotImplementedError
