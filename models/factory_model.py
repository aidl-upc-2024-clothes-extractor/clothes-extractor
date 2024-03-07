import torch.nn as nn
import segmentation_models_pytorch as smp
from models.discriminator import Discriminator
from trainer.cgan_trainer import CGANTrainerConfiguration

from trainer.unet_trainer import UnetTrainerConfiguration


def get_model(model_name: str):
    model_name = model_name.lower()
    if model_name.startswith("unet"):
        unet_params = model_name.split("-")
        if len(unet_params) == 3:
            unet_params.append(None)
        unet_params = [None if param == "none" else param for param in unet_params]

        model = smp.Unet(
            encoder_name=unet_params[1],
            encoder_weights=unet_params[2],
            decoder_attention_type=unet_params[3],
            in_channels=3,
            classes=3,
        )
        print(f"Using Unet:\n\t encoder_name={unet_params[1]}\n\t encoder_weights={unet_params[2]}\n\t decoder_attention_type={unet_params[3]}")
        scheduler = None
        if "onecyclelr" in unet_params:
            scheduler = "onecyclelr"
        return model, UnetTrainerConfiguration(model, scheduler), None
    if model_name.startswith("cgan"):
        cgan_params = model_name.split("-")
        if len(cgan_params) == 3:
            cgan_params.append(None)
        cgan_params = [None if param == "none" else param for param in cgan_params]

        model = smp.Unet(
            encoder_name=cgan_params[1],
            encoder_weights=cgan_params[2],
            decoder_attention_type=cgan_params[3],
            in_channels=6,
            classes=3,
        )
        discriminator = Discriminator()
        print(f"Using Unet:\n\t encoder_name={cgan_params[1]}\n\t encoder_weights={cgan_params[2]}\n\t decoder_attention_type={cgan_params[3]}")
        scheduler = None
        # if "onecyclelr" in cgan_params:
        #     scheduler = "onecyclelr"
        return model, CGANTrainerConfiguration(model, discriminator, scheduler), discriminator
    else:
        raise ValueError(f"Model {model_name} not supported.")