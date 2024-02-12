from dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

import torch
from src.model import Unet
from src.train import train_model
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from src.model_store import ModelStore
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
from tqdm import tqdm

import seaborn as sns
from scipy import stats
import pingouin as pg

def get_files(dataset_dir, dataset_mode):
    dataset_list = f'{dataset_mode}_pairs.txt'

    # load data list
    img_names = []
    with open(os.path.join(dataset_dir, dataset_list), 'r') as f:
        for line in f.readlines():
            img_name, c_name = line.strip().split()
            img_names.append(img_name)

    return img_names    

def calculate_ssim(model, device, dataset: ClothesDataset):
    model.eval()
    
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    index = 0
    acum = 0.0
    ssim_list = []
    for i in tqdm(range(len(dataset))):
    # for i in range(2):
        image = {}
        target = dataset[i]["target"].to(device)
        image["target"] = target
#        source = dataset[i]["composed_centered_mask_body"].to(device)
        source = dataset[i]["centered_mask_body"].to(device)
        source = source[None,:,:,:]
 
        output = model(source)
        output = output.detach()

        output = output.squeeze()
        image["output"] = output

        image_keys = ["target", "output"]
        ssim = ssim_calc(torch.unsqueeze(output.cpu(),0), torch.unsqueeze(target.cpu(),0))
        acum += ssim.item()
        ssim_list.append(ssim.item())
        # fig, axes = plt.subplots(1, len(image_keys))
        # for ax, key in zip(axes, image_keys):
        #     image[key] -= image[key].min()
        #     ax.imshow(image[key].cpu().permute(1, 2, 0))
        #     ax.axis('off')
        #     ax.set_title(key, rotation=90, fontsize=10)
        # plt.show()
    print(acum/len(dataset))
    return ssim_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")
    args = ArgumentParser(Config)
    cfg = args.parse_args()
    desired_cloth = cfg.predict_image 

    #cfg = Config()
    #cfg.load_height = 28
    #cfg.load_width = 28

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    model = Unet(in_channels=3, n_feat=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ms = ModelStore()
    reload_model = None
    if cfg.reload_model != "None" and  cfg.reload_model != "latest":
        reload_model = cfg.reload_model
    print(reload_model)
    
    model, optimizer, epoch, loss = ms.load_model(model, optimizer, model_name=reload_model)
    ssim_1 = []
    if cfg.predict_dataset == "train":
        ssim_1 = calculate_ssim(model, device, train_dataset)
    else:
        ssim_1 = calculate_ssim(model, device, test_dataset)
    with open("./ssim_3.txt", "w") as txt_file:
        for d in ssim_1:
            txt_file.write(str(d) + "\n")

if __name__ == '__main__':
    main()
