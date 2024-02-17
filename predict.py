import argparse
from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

import torch
#from models.unet import  Unet
import segmentation_models_pytorch as smp
from trainer.trainer import train_model
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import models.model_store as model_store
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.transforms as T


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

    test_dataset = ClothesDataset(cfg, "test")
    train_dataset = ClothesDataset(cfg, "train")

    test_dataloader = ClothesDataLoader(test_dataset, cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)
    train_dataloader = ClothesDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=cfg.dataloader_pin_memory)

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
        decoder_attention_type="scse"
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    #ms = ModelStore()
    reload_model = None
    if cfg.reload_model != None and cfg.reload_model != "latest":
        reload_model = cfg.reload_model
    print(reload_model)

    image = None
    if cfg.predict_dataset == "train":
        image = train_dataset[desired_cloth]
        model, optimizer, epoch, loss, val_loss = model_store.load_model(model, optimizer, reload_model)
        out = run_model_on_image(model, device, train_dataset, desired_cloth)
        #visualize_nn_output(out, device)
    else:
        image = test_dataset[desired_cloth]
        model, optimizer, epoch, loss, val_loss = model_store.load_model(model, optimizer, reload_model)
        out = run_model_on_image(model, device, test_dataset, desired_cloth)
        #visualize_nn_output(out, device)
        
    out = out.squeeze() 
    image["out"] = out

    # image_keys = ["img", "cloth", "cloth_mask", "predict", "agnostic_mask", "mask_body_parts", "mask_body", "centered_mask_body", "img_masked"]
    image_keys = ["target", "centered_mask_body", "out"]
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, gaussian_kernel=False, kernel_size=19).to(device)
    print(ssim(torch.unsqueeze(image["target"].to(device),0), torch.unsqueeze(image["out"].to(device),0)))

    fig, axes = plt.subplots(1, len(image_keys))
    transform = T.ToPILImage()

    for ax, key in zip(axes, image_keys):
        result = image[key] / 2 + 0.5
        for c in range(result.shape[0]):
            min_chn = result[c].min()
            if min_chn < 0.0:
                result[c] = result[c] - min_chn
        
        ax.imshow(transform(result))
        #ax.imshow(result.cpu().permute(1,2,0))
        ax.axis('off')
        ax.set_title(key, rotation=90, fontsize=10)
    target = image["target"] / 2 + 0.5
    out = image["out"] / 2 + 0.5
    print(ssim(torch.unsqueeze(target.to(device),0), torch.unsqueeze(out.to(device),0)))
    print(image["out"].shape)
    plt.show()

if __name__ == '__main__':
    main()
