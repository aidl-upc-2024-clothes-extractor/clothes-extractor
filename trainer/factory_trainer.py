

from abc import ABC
import string
from trainer.trainer import Trainer
from trainer.unet_trainer import UnetTrainer, UnetTrainerConfiguration


# Define a enum or similar with the unique names of the trainers
# This is useful to avoid typos and to have a single place to check for the names
class TrainerName:
    UNET = 'unet'
    # Add new trainers here

class TrainerConfiguration(ABC):
    name: string
    configuration: dict
    def __init__(self, name: string, configuration: dict):
        self.name = name
        self.configuration = configuration
    

def get_trainer(trainer_configuration:TrainerConfiguration) -> Trainer:
    """
    Returns the trainer object based on the configuration.
    """
    if isinstance(trainer_configuration, UnetTrainerConfiguration):
        return UnetTrainer(trainer_configuration)
    else:
        raise ValueError(f"Trainer {trainer_configuration['trainer']} not supported.")
