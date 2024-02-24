from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from trainer.trainer import LossTracker

class Logger(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_training(
        self,
        epoch: int,
        loss_tracker: LossTracker,
    ):
        pass

    @abstractmethod
    def log_images(
        self, epoch:int, train_images: torch.Tensor, val_images: torch.Tensor, train_target: torch.Tensor, val_target: torch.Tensor
    ):
        pass
