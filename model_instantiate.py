from config import Config
from models.unet import Unet
from models.model_store import ModelStore
import torch.optim as optim

def get_model(cfg: Config, device: str):
    learning_rate = cfg.learning_rate
    model = Unet(in_channels=3, n_feat=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    local_storer = ModelStore()
    reload_model = cfg.reload_model
    print(reload_model)
    if reload_model is not None and reload_model != "None":
        model, optimizer, epoch, loss = local_storer.load_model(model=model, optimizer=optimizer, model_name=reload_model)

    if reload_model is not None and reload_model == "latest":
        reload_model = None
        model, optimizer, epoch, loss = local_storer.load_model(model=model, optimizer=optimizer, model_name=reload_model)

    return model, optimizer, epoch, loss