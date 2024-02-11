from config import Config
from models.unet import Unet

def get_model(cfg: Config, device: str):
    device = cfg.device
    return Unet(in_channels=3, n_feat=32).to(device)