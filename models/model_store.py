import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from pathlib import Path

class ModelStore():
    """
    Class that stores and recovers a model to/from disk.
    Values stored and recovered are: model, optimizer, epoch, and loss.

    Usage:
        m = Model(...)
        o = optim(...)
        ms = ModelStorer() # We use the default folder value
        .........
        ms.save_model(m, o, epoch, loss, "mymodel") # It will create a file yyyymmdd_HHMM_e#_mymodel.pt
        .........
        m1 = Model(....)
        o1 = optim(....)
        m1, 01, epoch1, loss1 = ms.load_model(m1, o1) # it will read the most recent file
    """

    def __init__(self, path: str = "./model_checkpoints", model_name: str = "default_model_name"):
        """
        Initialize the ModelStore class.

        Args:
            path (str): The folder where all the checkpoints will be stored and used as a prefix in all operations.
                        Default is "./model_checkpoints".
            model_name (str): The default name for the model. Default is "default_model_name".
        """
        self.path = path
        self.model_name = model_name
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: nn.Module, optimizer: optim, epoch: int, loss: float):
        """
        Save a model to disk.

        Args:
            model (nn.Module): The model to be stored on disk.
            optimizer (optim): The optimizer state to be stored.
            epoch (int): The epoch number.
            loss (float): The current loss.
        """
        prefix = f'{datetime.now():%Y%m%d_%H%M}'
        filename = os.path.join(self.path, prefix + "_e" + str(epoch) + "_" + self.model_name + ".pt")
        saving_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(saving_dict, filename)

def load_model(self, model: nn.Module, optimizer: optim, model_name: str = None):
    """
    Load a model from disk.

    Args:
        model (nn.Module): The nn.Module object representing your neural network.
        optimizer (optim): The torch.optim object representing the optimizer.
        model_name (str): The name of the file to be read from disk. If not provided, the most recent file stored will be loaded.

    Returns:
        Tuple[nn.Module, optim, int, float]: The loaded model, optimizer, epoch, and loss.
    """
    loss = 0.0
    epoch = 0
    if model_name is None:
        p_check = Path(self.path)
        files = sorted(p_check.glob('*'))
        model_name = files[-1].name
    full_file_name = os.path.join(self.path,model_name)
    if Path(full_file_name).exists():
        checkpoint = torch.load(full_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    return model, optimizer, epoch, loss
