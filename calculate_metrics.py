import os
import json
from config import Config
from argparse_dataclass import ArgumentParser

import torch
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, SpatialCorrelationCoefficient, PeakSignalNoiseRatio, RelativeAverageSpectralError, ErrorRelativeGlobalDimensionlessSynthesis, LearnedPerceptualImagePatchSimilarity,MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms as T

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pingouin as pg

from tqdm import tqdm

from dataset.dataset import ClothesDataset, ClothesDataLoader, split_clothes_dataset
import models.sotre.model_store as model_store
from models.factory_model import get_model
from trainer.factory_trainer import get_trainer
import segmentation_models_pytorch as smp

def run_model_on_image(model, device, image):
    model.eval()
    
    image = image.to(device).unsqueeze(0)

    
    with torch.no_grad():
        output = model(image)

    return output
def get_files(dataset_dir, dataset_mode):
    dataset_list = f'{dataset_mode}_pairs.txt'

    # load data list
    img_names = []
    with open(os.path.join(dataset_dir, dataset_list), 'r') as f:
        for line in f.readlines():
            img_name, c_name = line.strip().split()
            img_names.append(img_name)

    return img_names    

def calculate_metrics(model, device, dataset: ClothesDataset, num_images: int = None):
    model.eval()
    
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_calc = PeakSignalNoiseRatio().to(device)
    ms_ssim_calc = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    metric_avg = {
        "ssim": 0.0,
        "psnr": 0.0,
        "ms_ssim": 0.0
    }
    metric_list = []
    if num_images is None:
        num_images = len(dataset)
    for i in tqdm(range(num_images)):
        metric = {
            "name": "",
            "ssim": 0.0,
            "psnr": 0.0,
            "ms_ssim": 0.0
        }
        image = dataset[i]
        metric["name"] = image["img_name"]
        target = image["target"]
        target = target / 2 + 0.5
        # print(f'Min Target: {target.min()}, Max Target: {target.max()}')

        source = image["centered_mask_body"]
        source = source / 2 + 0.5
        # print(f'Min Source: {source.min()}, Max Source: {source.max()}')
 
        output = run_model_on_image(model, device, image["centered_mask_body"])
        output = output.squeeze()
        output = torch.clamp(output / 2 + 0.5, 0.0, 1.0)
        # print(f'Min Output: {output.min()}, Max Output: {output.max()}')

        ssim = ssim_calc(torch.unsqueeze(output.cpu(),0), torch.unsqueeze(target.cpu(),0))
        metric["ssim"] = ssim.item()
        metric_avg["ssim"] += ssim.item()
        
        psnr = psnr_calc(torch.unsqueeze(output.cpu(),0), torch.unsqueeze(target.cpu(),0))
        metric["psnr"] = psnr.item()
        metric_avg["psnr"] += psnr.item()
        
        ms_ssim = ms_ssim_calc(torch.unsqueeze(output.cpu(),0), torch.unsqueeze(target.cpu(),0))
        metric["ms_ssim"] = ms_ssim.item()
        metric_avg["ms_ssim"] += ms_ssim.item()
        
        metric_list.append(metric)
    metric_avg["ssim"] = metric_avg["ssim"] / num_images
    metric_avg["psnr"] = metric_avg["psnr"] / num_images
    metric_avg["ms_ssim"] = metric_avg["ms_ssim"] / num_images
    return metric_avg, metric_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.device != "cuda":
        device = torch.device("mps")
    args = ArgumentParser(Config)
    cfg = args.parse_args()

    reload_model = cfg.reload_model
    reload_config = cfg.reload_config
    if reload_config is not None and reload_model is None:
        raise ValueError("reload_config is set but reload_model is not. Both must be set or none.")
    if reload_config is not None:
        old_cfg = model_store.load_previous_config(reload_model, device=device)
        print(f"Reading previous config from {reload_model}")
        for a in dir(old_cfg):
            if not a.startswith('__'):
                print(f"    {a}: {getattr(old_cfg, a)}")
        # Allow to overwrite the model name to support previous runs
        if cfg.model_name is None:
            old_cfg.model_name = cfg.model_name
        old_cfg.predict_image = cfg.predict_image
        old_cfg.predict_dataset = cfg.predict_dataset
        old_cfg.workers = 0
        
        cfg = old_cfg

    if cfg.model_name is None:
        raise ValueError("model-name must be set")

    full_dataset = ClothesDataset(cfg, "train", device="cpu")
    train_dataset, validation_dataset = split_clothes_dataset(full_dataset, [0.9, 0.1], generator=None)
    train_dataset.data_augmentation = False
    validation_dataset.data_augmentation = False

    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)
    validation_dataloader = ClothesDataLoader(validation_dataset, cfg.batch_size, num_workers=cfg.workers)

    model, trainer_configuration, discriminator = get_model(cfg.model_name)
    model.to(device)

    optimizer = None
    model, optimizer, epoch, loss, val_loss = model_store.load_model(model, path=reload_model, device=device)
    with torch.no_grad():
        num_images_remote = 16
        metrics_avg, metrics = calculate_metrics(model, device, train_dataloader.data_loader.dataset, num_images_remote)
        # print(json.dumps(metrics, indent=2))
        print(json.dumps(metrics_avg, indent=2))

    # with open("./ssim_3.txt", "w") as txt_file:
    #     for d in metrics:
    #         txt_file.write(str(d) + "\n")
    with torch.no_grad():
        num_images_remote = 16
        metrics_avg, metrics = calculate_metrics(model, device, validation_dataloader.data_loader.dataset, num_images_remote)
        # print(json.dumps(metrics, indent=2))
        print(json.dumps(metrics_avg, indent=2))
    # with open("./ssim_4.txt", "w") as txt_file:
    #     for d in metrics:
    #         txt_file.write(str(d) + "\n")

if __name__ == '__main__':
    main()

