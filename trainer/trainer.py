from abc import ABC, abstractmethod
from logging import Logger

from torch.nn import Module

from config import Config
from models.sotre.model_store import ModelStore
from models.sotre.wandb_store import WandbStore


class Trainer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_model(
        self,
        device,
        train_dataloader,
        val_dataloader,
        cfg: Config,
        logger: Logger,
        remote_model_store: WandbStore,
        local_model_store: ModelStore,
        start_from_epoch: int = 0,
    ) -> Module:
        pass
