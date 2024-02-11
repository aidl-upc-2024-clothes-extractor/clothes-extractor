import argparse
from argparse_dataclass import ArgumentParser
import logging
import os
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import wandb

from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from models.unet import Unet
import models.model_store as model_store
from trainer.trainer import train_model

from models.wandb_store import WandbStorer
from models.dummy_wandb_store import DummyWandbStorer
from metrics.wandb_logger import WandbLogger
from metrics.local_logger import LocalLogger


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
    args = ArgumentParser(Config)
    cfg = args.parse_args()

    # TODO: geet error level from config
    logger = logging.getLogger('clothes-logger')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
            and torch.device != "cuda"
        ):
            device = torch.device("mps")
    else:
        device = torch.device(cfg.device)
    
    logger.info('device: %s', device)
    logger.info('num_epochs: %s', cfg.num_epochs)
    logger.debug('current_path: %s', os.getcwd())
    logger.info('dataset_dir: %s', cfg.dataset_dir)

    print("Loading dataset...")
    dataset_device = cfg.dataset_device
    if dataset_device == "default": dataset_device = device
    test_dataset = ClothesDataset(cfg, "test", device=dataset_device)
    train_dataset = ClothesDataset(cfg, "train", device=dataset_device)

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)
    print("Done")

    model_store_unet_1 = model_store.ModelStore(model_name="smp.unet.efficientnet-b0.imagenet")
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    epoch = 0
    if cfg.reload_model is not None:
        print(reload_model)

        if reload_model == "latest":
            reload_model = None

        model, optimizer, epoch, loss = model_store.load_model(
            model=model, optimizer=optimizer, path=cfg.reload_model
        )
        epoch += 1
        print(f"Loaded model from ${cfg.reload_model} at epoch {epoch} with loss {loss}")

    # WANDB
    if cfg.dissable_wandb:
        wandb_storer = DummyWandbStorer()
        wandb_logger = LocalLogger()
    else:
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

    trained_model = train_model(
        optimizer=optimizer,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        cfg=cfg,
        logger=wandb_logger,
        remote_model_store=wandb_storer,
        local_model_store=model_store_unet_1,
        start_from_epoch=epoch,
    )

    out = run_model_on_image(model, device, train_dataset, 2)
    visualize_nn_output(out, device)

    if cfg.dissable_wandb is False:
        wandb_run.finish()

if __name__ == "__main__":
    main()
