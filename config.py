from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # General config
    batch_size: int = 64
    workers: int = 0
    dataset_dir: str = 'data/zalando-hd-resized'
    num_epochs: int = 30
    learning_rate: float = 0.0003
    load_height: int = 224 # Must be divisible by 32
    load_width: int = 224 # Must be divisible by 32
    dataloader_pin_memory: bool = False
    dataset_pairs_dir: str = 'data'

    # For development allow setting number of batches to not run the whole dataset
    max_batches: int = 0

    # Data Augmentation
    data_augmentation: bool = False
    horizontal_flip_prob: float = 0.5
    rotation_prob: float = 0.5
    rotation_angle: float = 10
    crop_prob: float = 0.25
    min_crop_factor: float = 0.65
    max_crop_factor: float = 0.92
    brightness: float = 0.15
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05
    color_jitter_prob: float = 0
    angle_prob: float = 0.2
    angle: float = 10

    # Other config
    device: str = None
    dataset_device: str = 'cpu'
    predict_image: int = 1 #  Image index to predict
    reload_model: str = None
    ssim_range: float = 1.0
    disable_wandb: bool = False
    model_name: str = "default"
    max_models_to_keep: int = None # Save only the best n models in the local disk
    predict_dataset: str = "test"


