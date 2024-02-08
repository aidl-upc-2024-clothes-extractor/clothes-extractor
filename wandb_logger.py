import torch.nn as nn
import wandb

from datetime import datetime


class WandbLogger:

    def __init__(
        self,
        model: nn.Module,
    ):
        wandb.login()
        self.run = wandb.init(project="clothes-extractor")
        self.run.name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch
        self.run.watch(models=model)

    def save_model(self, checkpoint_path):
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
        self.run.finish()

