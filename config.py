from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # General config
    batch_size: int = 64 #16
    workers: int = 0
    dataset_dir: str = 'data/zalando-hd-resized'
    num_epochs: int = 20
    learning_rate: float = 0.0002
    load_height: int = 1024
    load_width: int = 1024

    # For development allow setting number of batches to not run the whole dataset
    max_batches: int = 0

    # Data Augmentation
    data_augmentation: bool = False
    horizontal_flip_prob: float = 0.5
    rotation_prob: float = 0.5
    rotation_angle: float = 10
    crop_prob: float = 0.5
    min_crop_factor: float = 0.65
    max_crop_factor: float = 0.92
    brightness: float = 0.15
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05
    color_jitter_prob: float = 0
    angle_prob: float = 0.5
    angle: float = 10

    # Other config
    device: str = None
    predict_image: int = 1
    reload_model: str = None
    ssim_range: float = 1.0


