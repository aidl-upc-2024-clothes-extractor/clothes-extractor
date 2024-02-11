from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser
import logging
import os
from datetime import datetime

import torch
from models.unet import Unet
from trainer.trainer import train_model

import matplotlib.pyplot as plt
import numpy as np

import wandb
from models.wandb_store import WandbStorer
from metrics.wandb_logger import WandbLogger

def run_model_on_image(model, device, dataset, image_index):
    model.eval()

    image = dataset[image_index]
    image = image["centered_mask_body"].to(device).unsqueeze(0)


    with torch.no_grad():
        output = model(image)

    return output

def visualize_nn_output(output, device, image_index=0):
    output = output[image_index].squeeze().detach().cpu().numpy()

    if output.shape[0] in [3, 4]:  # RGB or RGBA
        output = np.transpose(output, (1, 2, 0))
    output -= output.min()
    
    plt.imshow(output)
    plt.show()

def main():
    # TODO: geet error level from config
    logger = logging.getLogger('clothes-logger')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")
    logger.info('device: %s', device)

    args = ArgumentParser(Config)
    cfg = args.parse_args()

    logger.info('num_epochs: %s', cfg.num_epochs)

    #print the python running  directory
    logger.debug('current_path: %s', os.getcwd())
    logger.info('dataset_dir: %s', cfg.dataset_dir)

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)

    model = Unet(in_channels=3, n_feat=32).to(device)

    # WANDB
    wandb.login()
    wandb_run = wandb.init(
        project="clothes-extractor",
        entity="clothes-extractor",
    )
    wandb_run.name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch
    wandb_run.watch(models=model) #, log=UtLiteral["gradients", "weights"])

    wandb_storer = WandbStorer(wandb_run)
    wandb_logger = WandbLogger(wandb_run)

    trained_model = train_model(model, device, train_dataloader, test_dataloader, cfg, wandb_logger, wandb_storer)
    out = run_model_on_image(model, device, train_dataset, 2)
    visualize_nn_output(out, device)

    wandb_run.finish

if __name__ == '__main__':
    main()
