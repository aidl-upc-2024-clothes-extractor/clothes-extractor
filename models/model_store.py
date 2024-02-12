import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from pathlib import Path

from config import Config

_DEFAULT_CHECKPOINT_PATH = "./model_checkpoints"

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

    def __init__(self, path: str = None, model_name: str = "default_model_name", wabdb_id:str = None):
        """
        Initialize the ModelStore class.

        Args:
            path (str): The folder where all the checkpoints will be stored and used as a prefix in all operations.
                        Default is "./model_checkpoints".
            model_name (str): The default name for the model. Default is "default_model_name".
        """
        if path is None:
            path = _DEFAULT_CHECKPOINT_PATH
        self.path = path
        self.model_name = model_name
        self.wabdb_id = wabdb_id
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)


    def save_model(self, cfg: Config,  model: nn.Module, optimizer: optim, epoch: int, loss: float, val_loss: float):
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
            'val_loss': val_loss,
            'wabdb_id': self.wabdb_id,
            'config': cfg
        }
        torch.save(saving_dict, filename)
        return filename

def load_previous_config(path: str = None):
    if path is None or path == "latest":
        p_check = Path(_DEFAULT_CHECKPOINT_PATH)
        files = sorted(p_check.glob('*'))
        path = files[-1].name
        full_file_name = os.path.join(_DEFAULT_CHECKPOINT_PATH, path)
    else:
        full_file_name = path
    if Path(full_file_name).exists():
        checkpoint = torch.load(full_file_name)
        config = checkpoint['config']

    else:
        raise FileNotFoundError(f"File {full_file_name} does not exist")
    return config


def load_previous_wabdb_id(path: str = None):
    if path is None:
        p_check = Path(_DEFAULT_CHECKPOINT_PATH)
        files = sorted(p_check.glob('*'))
        path = files[-1].name
        full_file_name = os.path.join(_DEFAULT_CHECKPOINT_PATH, path)
    else:
        full_file_name = path
    if Path(full_file_name).exists():
        checkpoint = torch.load(full_file_name)
        wabdb_id = checkpoint['wabdb_id']

    else:
        raise FileNotFoundError(f"File {full_file_name} does not exist")
    return wabdb_id

def load_model(model: nn.Module, optimizer: optim, path: str = None):
    """
    Load a model from disk.

    Args:
        model (nn.Module): The nn.Module object representing your neural network.
        optimizer (optim): The torch.optim object representing the optimizer.
        path (str): The name of the file to be read from disk. If not provided, the most recent file stored will be loaded.

    Returns:
        Tuple[nn.Module, optim, int, float]: The loaded model, optimizer, epoch, and loss.
    """
    loss = 0.0
    epoch = 0
    if path is None:
        p_check = Path(_DEFAULT_CHECKPOINT_PATH)
        files = sorted(p_check.glob('*'))
        path = files[-1].name
        full_file_name = os.path.join(_DEFAULT_CHECKPOINT_PATH, path)
    else:
        full_file_name = path
    if Path(full_file_name).exists():
        checkpoint = torch.load(full_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']

    else:
        raise FileNotFoundError(f"File {full_file_name} does not exist")
    return model, optimizer, epoch, loss, val_loss

