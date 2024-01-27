import argparse
from dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

import torch
from src.model import Unet
from src.train import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")

    args = ArgumentParser(Config)
    cfg = args.parse_args()
    print(cfg.batch_size)

    cfg = Config()
    cfg.load_height = 28
    cfg.load_width = 28

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    model = Unet(in_channels=3, n_feat=32).to(device)

    trained_model = train_model(model, device, train_dataloader, test_dataloader, cfg.num_epochs, cfg.learning_rate, cfg.max_batches)


if __name__ == '__main__':
    main()
