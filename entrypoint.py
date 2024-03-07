from argparse_dataclass import ArgumentParser
import logging
import os

import torch


import matplotlib.pyplot as plt
import numpy as np
import wandb

from dataset.dataset import ClothesDataset, ClothesDataLoader, split_clothes_dataset
from config import Config
from models.unet import Unet
import models.sotre.model_store as model_store
from models.factory_model import get_model
from trainer.factory_trainer import get_trainer

from models.sotre.wandb_store import WandbStore
from models.sotre.dummy_wandb_store import DummyWandbStorer
from metrics.wandb_logger import WandbLogger
from metrics.local_logger import LocalLogger
from torch import optim


def run_model_on_image(model, device, dataset, image_index):
    model.eval()

    image = dataset[image_index]
    image = image["centered_mask_body"].to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    return output


def visualize_nn_output(output, device, image_index=0):
    output = output[image_index].squeeze().detach().cpu().numpy()

    if output.shape[0] in [3, 4]:  # RGB or RGBA
        output = np.transpose(output, (1, 2, 0))
    output -= output.min()

    plt.imshow(output)
    plt.show()


def main():
    args = ArgumentParser(Config)
    cfg:Config = args.parse_args()

    reload_model = cfg.reload_model
    reload_config = cfg.reload_config
    if reload_config is not None and reload_model is None:
        raise ValueError("reload_config is set but reload_model is not. Both must be set or none.")
    if reload_config is not None:
        old_cfg = model_store.load_previous_config(reload_model)
        print(f"Reading previous config from {reload_model}")
        for a in dir(old_cfg):
            if not a.startswith('__'):
                print(f"    {a}: {getattr(old_cfg, a)}")
        # Allow to overwrite the model name to support previous runs
        if cfg.model_name is None:
            old_cfg.model_name = cfg.model_name

        new_epochs = cfg.num_epochs
        cfg = old_cfg
        if cfg.num_epochs < new_epochs:
            cfg.num_epochs = new_epochs

    if cfg.model_name is None:
        raise ValueError("model-name must be set")
            
    # TODO: get error level from config
    logger = logging.getLogger("clothes-logger")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
            and torch.device != "cuda"
        ):
            device = torch.device("mps")
    else:
        device = torch.device(cfg.device)

    logger.info("device: %s", device)
    logger.info("num_epochs: %s", cfg.num_epochs)
    logger.debug("current_path: %s", os.getcwd())
    logger.info("dataset_dir: %s", cfg.dataset_dir)

    print("Loading dataset...")
    dataset_device = cfg.dataset_device
    if dataset_device == "default":
        dataset_device = device

    full_dataset = ClothesDataset(cfg, "train", device=dataset_device)
    #test_dataset = ClothesDataset(cfg, "test", device=dataset_device)
    train_dataset, validation_dataset = split_clothes_dataset(full_dataset, [0.9, 0.1], generator=None)

    # test_dataloader = ClothesDataLoader(
    #     test_dataset,
    #     cfg.batch_size,
    #     num_workers=cfg.workers,
    #     pin_memory=cfg.dataloader_pin_memory,
    # )
    validation_dataloader = ClothesDataLoader(
        validation_dataset,
        cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=cfg.dataloader_pin_memory,
    )
    train_dataloader = ClothesDataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=cfg.dataloader_pin_memory,
    )
    print("Done")

    model, trainer_configuration, discriminator = get_model(cfg.model_name)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if discriminator is None:
        optimizerD = None
    else:
        discriminator.to(device)
        optimizerD = optim.Adam(discriminator.parameters(), lr=cfg.discriminator_learning_rate, betas=(0.5, 0.999))
    trainer_configuration.optimizer = optimizer
    trainer_configuration.optimizerD = optimizerD

    epoch = 0
    resume = cfg.no_resume_wandb is False
    if reload_model is not None:
        model, optimizer, epoch, loss, val_loss = model_store.load_model(
            model=model, optimizer=optimizer, path=reload_model,
            discriminator=discriminator, optimizerD=optimizerD
        )
        epoch += 1
        print(f"Loaded model from ${reload_model} at epoch {epoch}/{cfg.num_epochs}. test_loss={loss} val_loss={val_loss}")

    # WANDB
    wabdb_id = cfg.previous_wandb_id
    wandb_run = None
    if cfg.disable_wandb:
        wandb_store = DummyWandbStorer()
        wandb_logger = LocalLogger()
    else:
        if resume and reload_model is not None:
            wabdb_id = model_store.load_previous_wabdb_id(reload_model)
        if wabdb_id is not None:
            resume = True
        wandb.login()
        wandb_run_name = cfg.model_name
        print(f'Starting wandb run with name {wandb_run_name} and id {wabdb_id} and resume={resume}')
        wandb_run = wandb.init(
            project="clothes-extractor",
            entity="clothes-extractor",
            id=wabdb_id,
            name=wandb_run_name,
            resume=resume
        )
        wabdb_id = wandb_run.id

        # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch
        wandb_run.watch(models=model)  # , log=UtLiteral["gradients", "weights"])

        wandb_store = WandbStore(wandb_run)
        wandb_logger = WandbLogger(wandb_run)

    local_model_store = model_store.ModelStore(
        model_name=cfg.model_name,
        wabdb_id=wabdb_id,
        max_models_to_keep=cfg.max_models_to_keep
    )

    trainer = get_trainer(trainer_configuration)

    trained_model = trainer.train_model(
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        cfg=cfg,
        logger=wandb_logger,
        remote_model_store=wandb_store,
        local_model_store=local_model_store,
        start_from_epoch=epoch,
    )

    out = run_model_on_image(trained_model, device, train_dataset, 2)
    visualize_nn_output(out, device)

    if cfg.disable_wandb is False:
        wandb_run.finish()


if __name__ == "__main__":
    main()
