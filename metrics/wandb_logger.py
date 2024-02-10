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
            train_acc_avg: np.ndarray,
            val_acc_avg: np.ndarray,
            fig: plt.Figure,
    ):
        raise NotImplementedError

