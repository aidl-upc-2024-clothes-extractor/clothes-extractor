import torch.nn as nn
import wandb


class WandbStore:

    def __init__(
        self,
        wandb_run,
    ):
        self.run = wandb_run

    def save_model(self, checkpoint_path):
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path, self.run.name)
        self.run.log_artifact(artifact, aliases=[self.run.name])

    def restore_model(self):
        wandb.use_artifact()
        #best_model = wandb.restore('model.h5', run_path="lavanyashukla/save_and_restore/10pr4joa")
