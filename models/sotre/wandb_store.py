import torch.nn as nn
import wandb
import argparse


class WandbStore:

    def __init__(
        self,
        wandb_run,
    ):
        self.run = wandb_run

    def save_model(self, checkpoint_path):
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(checkpoint_path, self.run.name)
        self.run.log_artifact(artifact, aliases=[self.run.name])

    def restore_model(self):
        wandb.use_artifact()
        #best_model = wandb.restore('model.h5', run_path="lavanyashukla/save_and_restore/10pr4joa")


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, default=None, required=True)
    parser.add_argument("-n", "--name", type=str, default=None, required=True)

    opt = parser.parse_args()
    return opt


def upload_checkpoint(opt):
    wandb_run = wandb.init(
        project="clothes-extractor",
        entity="clothes-extractor",
    )
    artifact = wandb.Artifact(name=opt.name, type="model")
    artifact.add_file(opt.file)
    wandb_run.log_artifact(artifact)


if __name__ == "__main__":
    opt = get_opt()
    upload_checkpoint(opt)
