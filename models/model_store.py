import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from pathlib import Path

# Class that store and recovers a model to/from disk
# Values stored and recovered are: model, optimizer, epoch and loss
# Usage:
#   m = Model(...)
#   o = optim(...)
#   ms = ModelStorer() # We use the default folder value
#   .........
#   ms.save_model(m, o, epoch, loss, "mymodel") # It will create a file yyyy_mm_dd_HH_MM_SS_mymodel.pt
#   .........
#   m1 = Model(....)
#   o1 = optim(....)
#   m1, 01, epoch1, loss1 = ms.load_model(m1, o1) # it will read the most recent file

class ModelStore():
    # Intializing the class. Defaulta path in case you don't send anyone. 
    # path: is the folder where all the checkpoints will be stored and will be used as prefix in all operations
    def __init__(self, path: str = "./model_checkpoints"):
        self.path = path
        # In case the path that you indicates doesn't exists, the object will create it.
        # You don't need to create all the folders in the path, the class will create all the structure
        p = Path(path)
        if not p.exists():
            # Create all the path with all the structure
            p.mkdir(parents=True, exist_ok=True)

    # Save a model in the disk
    # model: Model to be stored on disk
    # optimizer: When you store a model you need to store the state of the optimizer
    # epoch: epoch number
    # loss: current loss
    # model_name: the name you want for your model, the file will contain it. Default name in place
    # The file where the model will be stored, it will contains the date and time in the name in the format:
    # yyyy_mm_dd_HH_MM_SS_<<Provided model name>>.pt
    def save_model(self, model: nn.Module,  optimizer: optim, epoch: int, loss: float, model_name: str = "default_model_name"):
        # we create a prefix with current datetime
        prefix = f'{datetime.now():%Y_%m_%d_%H_%M_%S}'
        # Prepare the full filename
        filename = os.path.join(self.path, prefix + "_" + model_name + ".pt")
        # Prepare the model information
        saving_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(saving_dict, filename)
    # Load a model from disk. In order to read properly the model and optimizer, they have to be created previously and send as parameters
    # model: nn.Module object representing your nn. It has to have been instantiated with exactly the same model that is in the disk
    # optimizer: torch.optim object representing the optimizer. It has to be the same that it is in the disk
    # model_name: Name of the file to be read from disk, in case no argument is passed the most recent file stored will be loaded
    #       the filename has the format yyyy_mm_dd_HH_MM_SS_<<Provided model name>>.pt in case you don't send the name you want to read
    #       the file with the latest yyyy_mm_dd_HH_MM_SS will be loaded
    def load_model(self, model: nn.Module,  optimizer: optim, model_name: str = None):
        loss = 0.0
        epoch = 0
        if model_name is None:
            # In case no filename is provided the most recent file will be loaded
            p_check = Path(self.path)
            # Get the list of files in the model folder
            files = sorted(p_check.glob('*'))
            # Once the list is sorted from older to newer we take the newer (last possition of array)
            model_name = files[-1].name
        full_file_name = os.path.join(self.path,model_name)
        if Path(full_file_name).exists():
            checkpoint = torch.load(full_file_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            # Don't forget to assign the returned values to your model and optimizer
        return model, optimizer, epoch, loss