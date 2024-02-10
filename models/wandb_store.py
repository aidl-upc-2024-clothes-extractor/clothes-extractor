import torch.nn as nn
import wandb


class WandbStorer:

    def __init__(
        self,
        wandb_run,
    ):
        self.run = wandb_run

    def save_model(self, checkpoint_path):
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
