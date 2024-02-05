import argparse
from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser
import os

import torch
from models.unet import Unet
from trainer.trainer import train_model

import matplotlib.pyplot as plt
import numpy as np

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")

    args = ArgumentParser(Config)
    cfg = args.parse_args()
    print(cfg.num_epochs)
    #cfg = Config()
    #cfg.load_height = 28
    #cfg.load_width = 28

    #print the python running  directory
    print(os.getcwd())
    print(cfg.dataset_dir)

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    model = Unet(in_channels=3, n_feat=32).to(device)

    trained_model = train_model(model, device, train_dataloader, test_dataloader, cfg.num_epochs, cfg.learning_rate, cfg.max_batches, cfg.reload_model, cfg.ssim_range)
    out = run_model_on_image(model, device, train_dataset, 2)
    visualize_nn_output(out, device)

if __name__ == '__main__':
    main()
