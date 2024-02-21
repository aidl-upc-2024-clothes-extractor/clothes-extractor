
from trainer.trainer import Trainer
from trainer.trainer_configuration import TrainerConfiguration
from trainer.unet_trainer import UnetTrainer, UnetTrainerConfiguration
    

def get_trainer(trainer_configuration:TrainerConfiguration) -> Trainer:
    """
    Returns the trainer object based on the configuration.
    """
    if isinstance(trainer_configuration, UnetTrainerConfiguration):
        return UnetTrainer(trainer_configuration)
    else:
        raise ValueError(f"Trainer {trainer_configuration['trainer']} not supported.")
