from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # General config
    batch_size: int = 32
    workers: int = 0
    dataset_dir: str = 'data/zalando-hd-resized'
    num_epochs: int = 500
    learning_rate: float = 0.0003
    discriminator_learning_rate: float = 0.0002
    load_height: int = 256 # Must be divisible by 32
    load_width: int = 256 # Must be divisible by 32
    dataloader_pin_memory: bool = False
    dataset_pairs_dir: str = 'data'
    model_name: str = None

    # For development allow setting number of batches to not run the whole dataset
    max_batches: int = 0

    # Data Augmentation
    data_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    crop_prob: float = 0
    min_crop_factor: float = 0.65
    max_crop_factor: float = 0.92
    brightness: float = 0.15
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05
    color_jitter_prob: float = 0
    angle_prob: float = 0.2
    rotation_angle: float = 5

    # Other config
    device: str = None
    dataset_device: str = 'cpu'
    predict_image: int = 1  # Image index to predict
    reload_model: str = None
    reload_config: bool = None
    ssim_range: float = 1.0
    disable_wandb: bool = False
    no_resume_wandb: bool = False
    previous_wandb_id: str = None
    checkpoint_save_frequency: int = 1
    wandb_save_checkpoint: bool = True
    model_name: str = "default"
    max_models_to_keep: int = None  # Save only the best n models in the local disk
    predict_dataset: str = "test"


