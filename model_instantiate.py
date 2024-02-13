from config import Config
from models.unet import Unet
from models.model_store import ModelStore
import torch.optim as optim
from models.discriminator import Discriminator

def get_model(cfg: Config, device: str):
    learning_rate = cfg.learning_rate
    model = Unet(in_channels=3, n_feat=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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