import torch.nn as nn
import wandb


class DummyWandbStorer:

    def __init__(self):
        pass

    def save_model(self, checkpoint_path):
        pass
