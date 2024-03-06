import torch.nn as nn
import segmentation_models_pytorch as smp
from models.discriminator import Discriminator, ConditionalFCCGANDiscriminator, DiscriminatorFC
from trainer.cgan_trainer import CGANTrainerConfiguration
from trainer.pix2pix_trainer import Pix2PixTrainerConfiguration

from trainer.unet_trainer import UnetTrainerConfiguration


def get_model(model_name: str):
    model_name = model_name.lower()
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
        print(f"Using Unet:\n\t encoder_name={unet_params[1]}\n\t encoder_weights={unet_params[2]}\n\t decoder_attention_type={unet_params[3]}")
        scheduler = None
        if "onecyclelr" in unet_params:
            scheduler = "onecyclelr"
        return model, UnetTrainerConfiguration(model, scheduler), None
    if model_name.startswith("cgan"):
        cgan_params = model_name.split("-")
        if len(cgan_params) == 3:
            cgan_params.append(None)
        cgan_params = [None if param == "none" else param for param in cgan_params]

        model = smp.Unet(
            encoder_name=cgan_params[1],
            encoder_weights=cgan_params[2],
            decoder_attention_type=cgan_params[3],
            activation="tanh",
            in_channels=3,
            classes=3,
            aux_params={
                "dropout": 0.2,
                "classes": 3,
                "activation": "tanh"
            }
        )
        discriminator = DiscriminatorFC()
        print(f"Using Unet:\n\t encoder_name={cgan_params[1]}\n\t encoder_weights={cgan_params[2]}\n\t decoder_attention_type={cgan_params[3]}")
        scheduler = None
        # if "onecyclelr" in cgan_params:
        #     scheduler = "onecyclelr"
        return model, CGANTrainerConfiguration(model, discriminator, scheduler), discriminator
    if model_name.startswith("pix2pix"):
        from pix2pix.models.pix2pix_model import Pix2PixModel

        model = Pix2PixModel(Pix2PixDefaultOptions())
        print(f"Using Pix2Pix ")
        scheduler = None
        # if "onecyclelr" in cgan_params:
        #     scheduler = "onecyclelr"
        return model, Pix2PixTrainerConfiguration(model), None
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
class Pix2PixDefaultOptions:
    def __init__(self):
        self.display_freq = 400
        self.display_ncols = 4
        self.display_id = 1
        self.display_server = "http://localhost"
        self.display_env = 'main'
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False  # Since action='store_true', default should be False
        self.save_latest_freq = 5000
        self.save_epoch_freq = 5
        self.save_by_iter = False  # Since action='store_true', default should be False
        self.continue_train = False  # Since action='store_true', default should be False
        self.epoch_count = 1
        self.phase = 'train'
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.gan_mode = 'lsgan'
        self.pool_size = 50
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.isTrain = True  # Training mode by default
        self.dataroot = None  # Required field, no default provided
        self.name = 'experiment_name'
        self.gpu_ids = 'cuda'
        self.checkpoints_dir = './checkpoints'
        self.model = 'cycle_gan'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.norm = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = False  # Since action='store_true', default should be False
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'
        self.serial_batches = False  # Since action='store_true', default should be False
        self.num_threads = 4
        self.batch_size = 1
        self.load_size = 286
        self.crop_size = 256
        self.max_dataset_size = float("inf")
        self.preprocess = 'resize_and_crop'
        self.no_flip = False  # Since action='store_true', default should be False
        self.display_winsize = 256
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False  # Since action='store_true', default should be False
        self.suffix = ''
        self.use_wandb = False  # Since action='store_true', default should be False
        self.wandb_project_name = 'CycleGAN-and-pix2pix'
        self.initialized = True
