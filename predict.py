from dataset.dataset import ClothesDataset, ClothesDataLoader
from config import Config
from argparse_dataclass import ArgumentParser

import torch
#from models.unet import  Unet
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import models.sotre.model_store as model_store
from torchmetrics.image import StructuralSimilarityIndexMeasure, SpatialCorrelationCoefficient, PeakSignalNoiseRatio, RelativeAverageSpectralError, ErrorRelativeGlobalDimensionlessSynthesis, LearnedPerceptualImagePatchSimilarity,MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms as T
from models.factory_model import get_model
from trainer.factory_trainer import get_trainer


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
    cfg:Config = args.parse_args()

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

    desired_cloth = cfg.predict_image 

    used_dataset = ClothesDataset(cfg, cfg.predict_dataset)

    model, trainer_configuration, discriminator = get_model(cfg.model_name)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    optimizerD = None
    discriminator = None
    trainer_configuration.optimizer = optimizer
    trainer_configuration.optimizerD = optimizerD
    if reload_model is not None:
        model, optimizer, epoch, loss, val_loss = model_store.load_model(
            model=model, optimizer=optimizer, path=reload_model,
            discriminator=discriminator, optimizerD=optimizerD, device=device
        )
        epoch += 1
        print(f"Loaded model from ${reload_model} at epoch {epoch}/{cfg.num_epochs}. test_loss={loss} val_loss={val_loss}")
    model.to(device)


    image = used_dataset[desired_cloth]
    out = run_model_on_image(model, device, used_dataset, desired_cloth)
    out = out.squeeze() 
    image["out"] = out

    # image_keys = ["img", "cloth", "cloth_mask", "predict", "agnostic_mask", "mask_body_parts", "mask_body", "centered_mask_body", "img_masked"]
    # image_keys = ["target", "centered_mask_body", "out"]
    # fig, axes = plt.subplots(1, len(image_keys))
    # transform = T.ToPILImage()

    # for ax, key in zip(axes, image_keys):
    #     result = image[key] / 2 + 0.5
    #     result = torch.clamp(result, 0, 1)
    #     # for c in range(result.shape[0]):
    #     #     min_chn = result[c].min()
    #     #     if min_chn < 0.0:
    #     #         result[c] = result[c] - min_chn
        
    #     ax.imshow(transform(result))
    #     #ax.imshow(result.cpu().permute(1,2,0))
    #     ax.axis('off')
    #     ax.set_title(key, rotation=90, fontsize=10)

    target = image["target"] / 2 + 0.5
    source = image["centered_mask_body"] / 2 + 0.5
    out = torch.clamp(image["out"] / 2 + 0.5, 0, 1)
    print(f'Min Target: {target.min()}, Max Target: {target.max()}, Min Out: {out.min()}, Max Out: {out.max()}')

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    print(f'SSIM pred-targets: {ssim(torch.unsqueeze(out.to(device),0), torch.unsqueeze(target.to(device),0))}' +
          f' - SSIM Source-Target {ssim(torch.unsqueeze(source.to(device),0), torch.unsqueeze(target.to(device),0))}' +
          f' - SSIM Target-Target {ssim(torch.unsqueeze(target.to(device),0), torch.unsqueeze(target.to(device),0))}'
          )

    # vif = VisualInformationFidelity().to(device) #No sirve, el blanco sale alto y el negro bajo
    # print(vif(torch.unsqueeze(out.to(device),0), torch.unsqueeze(target.to(device),0)))

    psnr = PeakSignalNoiseRatio().to(device)
    print(f'PSNR pred-targets: {psnr(torch.unsqueeze(out.to(device),0), torch.unsqueeze(target.to(device),0))}' +
          f' - PSNR Source-Target {psnr(torch.unsqueeze(source.to(device),0), torch.unsqueeze(target.to(device),0))}' +
          f' - PSNR Target-Target {psnr(torch.unsqueeze(target.to(device),0), torch.unsqueeze(target.to(device),0))}'
          )

    ergas = ErrorRelativeGlobalDimensionlessSynthesis()
    print(f'ERGAS pred-targets: {torch.round(ergas(torch.unsqueeze(out.to(device),0), torch.unsqueeze(target.to(device),0)))}' + 
          f' - ERGAS Source-Target {torch.round(ergas(torch.unsqueeze(source.to(device),0), torch.unsqueeze(target.to(device),0)))}' + 
          f' - ERGAS Target-Target {torch.round(ergas(torch.unsqueeze(target.to(device),0), torch.unsqueeze(target.to(device),0)))}'
          )

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    print(f'LPIPS pred-targets: {lpips(torch.unsqueeze(out,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}' + 
          f' - LPIPS Source-Target {lpips(torch.unsqueeze(source,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}' +
          f' - LPIPS Target-Target {lpips(torch.unsqueeze(target,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
          )
    
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, normalize='relu')
    print(f'MS SSIM pred-targets: {ms_ssim(torch.unsqueeze(out,0).to("cpu"),torch.unsqueeze(target,0).to("cpu"))}'+
          f' - MS SSIM Source-Target {ms_ssim(torch.unsqueeze(source,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}' +
          f' - MS SSIM Target-Target {ms_ssim(torch.unsqueeze(target,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
          )

    # Da un Nan    
    # rase = RelativeAverageSpectralError()
    # print(f'RASE pred-targets: {rase(torch.unsqueeze(out,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}' + 
    #       f' - RASE Source-Target {rase(torch.unsqueeze(source,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
    #       f' - RASE Target-Target {rase(torch.unsqueeze(target,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
    #       )
    
    scc = SpatialCorrelationCoefficient()
    print(f'SCC pred-targets: {scc(torch.unsqueeze(out,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}' + 
          f' - SCC Source-Target {scc(torch.unsqueeze(source,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
          f' - SCC Target-Target {scc(torch.unsqueeze(target,0).to("cpu"), torch.unsqueeze(target,0).to("cpu"))}'
          )

    #plt.show()

if __name__ == '__main__':
    main()
