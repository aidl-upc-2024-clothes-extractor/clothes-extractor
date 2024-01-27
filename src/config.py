from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # General config
    batch_size: int = 64
    workers: int = 1
    dataset_dir: str = '../datasets/zalando-hd-resized'
    dataset_mode: str = 'test'
    num_epochs: int = 5
    learning_rate: float = 0.0002
    num_classes: int = 10
    num_input_channels: int = 1
    num_val_samples: int = 25

    # Data Augmentation
    load_height: int = 1024
    load_width: int = 1024
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
    color_jitter_prob: float = 1
    angle_prob: float = 0.5
    angle: float = 10


