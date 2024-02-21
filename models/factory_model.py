import torch.nn as nn
import segmentation_models_pytorch as smp

from trainer.unet_trainer import UnetTrainerConfiguration


def get_model(model_name: str):
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
        return model, UnetTrainerConfiguration(model)

    else:
        raise ValueError(f"Model {model_name} not supported.")