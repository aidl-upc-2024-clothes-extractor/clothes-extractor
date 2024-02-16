import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

from metrics.logger import Logger

from datetime import datetime


class LocalLogger(Logger):

    def __init__(self):
        pass

    def log_training(
            self,
            epoch: int,
            train_loss_avg: np.ndarray,
            val_loss_avg: np.ndarray,
    ):
        print(f"epoch {epoch} train_loss {np.mean(train_loss_avg)} val_loss {np.mean(val_loss_avg)}")

