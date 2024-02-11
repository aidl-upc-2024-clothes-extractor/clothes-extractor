import argparse
from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

import torch
from models.unet import  Unet
from trainer.trainer import train_model
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from models.model_store import ModelStore
from torchmetrics.image import StructuralSimilarityIndexMeasure


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

    plt.imshow(output)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")
    args = ArgumentParser(Config)
    cfg = args.parse_args()
    print(cfg.num_epochs)
    desired_cloth = cfg.predict_image 

    #cfg = Config()
    #cfg.load_height = 28
    #cfg.load_width = 28

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)

    model = Unet(in_channels=3, n_feat=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ms = ModelStore()
    reload_model = None
    if cfg.reload_model != "None" and  cfg.reload_model != "latest":
        reload_model = cfg.reload_model
    print(reload_model)
    
    image = train_dataset[desired_cloth]
    model, optimizer, epoch, loss = ms.load_model(model, optimizer, model_name=reload_model)
    out = run_model_on_image(model, device, train_dataset, desired_cloth)
    #visualize_nn_output(out, device)
        
    out = out.squeeze() 
    # print(out.min())
    # out -= out.min()
    #out /= out.max()
    image["out"] = out

    # image_keys = ["img", "cloth", "cloth_mask", "predict", "agnostic_mask", "mask_body_parts", "mask_body", "centered_mask_body", "img_masked"]
    image_keys = ["target", "centered_mask_body", "cloth_mask",  "out"]
    fig, axes = plt.subplots(1, len(image_keys))

    for ax, key in zip(axes, image_keys):
        image[key] -= image[key].min()
        ax.imshow(image[key].cpu().permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(key, rotation=90, fontsize=10)
    print(image["out"].shape)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    print(ssim(torch.unsqueeze(image["out"].to(device),0), torch.unsqueeze(image["target"].to(device),0)))
    plt.show()

if __name__ == '__main__':
    main()
