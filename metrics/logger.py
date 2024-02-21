from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt


class Logger(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        ssim: np.ndarray,
        perceptual: np.ndarray,
    ):
        pass

    @abstractmethod
    def log_images(
        self, epoch: int, train_images: torch.Tensor, val_images: torch.Tensor
    ):
        pass
