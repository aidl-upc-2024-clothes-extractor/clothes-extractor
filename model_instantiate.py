from config import Config
from models.unet import Unet
from models.model_store import ModelStore
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
    discriminator = Discriminator(ngpu=1).to(device)
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    local_storer = ModelStore()
    reload_model = cfg.reload_model
    print(reload_model)
    if reload_model is not None and reload_model != "None":
        model, optimizer, epoch, loss = local_storer.load_model(model=model, optimizer=optimizer, discriminator=discriminator, optimizerD=optimizerD, model_name=reload_model)
    elif reload_model is not None and reload_model == "latest":
        reload_model = None
        model, optimizer, epoch, loss = local_storer.load_model(model=model, optimizer=optimizer, discriminator=discriminator, optimizerD=optimizerD, model_name=reload_model)
    else:
        epoch = 0
        loss = 0

    return model, optimizer, discriminator, optimizerD, epoch, loss