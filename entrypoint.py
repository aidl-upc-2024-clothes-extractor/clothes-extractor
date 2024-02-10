import argparse
from argparse_dataclass import ArgumentParser
import os
from tqdm.auto import tqdm

import torch
import torch.optim as optim
import segmentation_models_pytorch as smp

from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from models.unet import Unet
import models.model_store as model_store
from trainer.trainer import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and torch.device != "cuda"
    ):
        device = torch.device("mps")

    args = ArgumentParser(Config)
    cfg = args.parse_args()
    print(cfg.batch_size)


    print("Loading dataset...")
    test_dataset = ClothesDataset(cfg, "test", device=device)
    train_dataset = ClothesDataset(cfg, "train", device=device)

    test_dataloader = ClothesDataLoader(
        test_dataset, cfg.batch_size, num_workers=cfg.workers
    )
    train_dataloader = ClothesDataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers
    )

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
    if cfg.continue_from:
        model, optimizer, epoch, loss = model_store.load_model(
            model=model, optimizer=optimizer, path=cfg.continue_from
        )
        print(f"Loaded model from ${cfg.continue_from} at epoch {epoch} with loss {loss}")

    trained_model = train_model(
        optimizer=optimizer,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=cfg.num_epochs,
        max_batches=cfg.max_batches,
        model_store=model_store_unet_1,
        start_from_epoch=epoch,
    )


if __name__ == "__main__":
    main()
