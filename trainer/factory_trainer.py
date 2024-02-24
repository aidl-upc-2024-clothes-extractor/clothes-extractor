
from trainer.cgan_trainer import CGANTrainer, CGANTrainerConfiguration
from trainer.trainer import Trainer
from trainer.trainer_configuration import TrainerConfiguration
from trainer.unet_trainer import UnetTrainer, UnetTrainerConfiguration
    

def get_trainer(trainer_configuration: TrainerConfiguration) -> Trainer:
    if trainer_configuration.optimizer is None:
        raise ValueError("Trainer configuration must have an optimizer.")
    
    if isinstance(trainer_configuration, UnetTrainerConfiguration):
        return UnetTrainer(trainer_configuration)
    if isinstance(trainer_configuration, CGANTrainerConfiguration):
        return CGANTrainer(trainer_configuration)
    else:
        raise ValueError(f"Trainer {trainer_configuration['trainer']} not supported.")
