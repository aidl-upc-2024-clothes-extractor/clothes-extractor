from config import Config
from models.unet import Unet
import torch.optim as optim
from models.discriminator import Discriminator
import segmentation_models_pytorch as smp

def get_model(cfg: Config, device: str):
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
        decoder_attention_type="scse"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    discriminator = Discriminator().to(device)
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    return model, optimizer, discriminator, optimizerD