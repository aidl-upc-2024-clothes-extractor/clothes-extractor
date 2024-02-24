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

    def __init__(self, path: str = None, model_name: str = "default_model_name", max_models_to_keep: int = None, wabdb_id:str = None):
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
        self.max_models_to_keep = max_models_to_keep
        self.models_saved = []
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    def save_model(self, cfg: Config, model: nn.Module,  optimizer: optim, discriminator: nn.Module, optimizerD: optim, epoch: int, loss: float, val_loss:float, model_name: str = "default_model_name"):
        """
        Save a model to disk.

        """
        prefix = f'{datetime.now():%Y%m%d_%H%M}'
        filename = os.path.join(self.path, prefix + "_e" + str(epoch) + "_" + self.model_name + ".pt")
        saving_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'loss': loss,
            'val_loss': val_loss,
            'wabdb_id': self.wabdb_id,
            'config': cfg
        }
        torch.save(saving_dict, filename)
        if self.max_models_to_keep is not None:
            self.models_saved.sort(key=lambda x: x[1])
            if len(self.models_saved) + 1 > self.max_models_to_keep:
                os.remove(self.models_saved.pop(-1)[0])

            self.models_saved.append((filename, loss))

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

def load_model(model: nn.Module,  optimizer: optim, discriminator: nn.Module, optimizerD: optim, path: str = None):
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
        if 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'optimizerD_state_dict' in checkpoint:
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']
    else:
        raise FileNotFoundError(f"File {full_file_name} does not exist")
    return model, optimizer, epoch, loss, val_loss
