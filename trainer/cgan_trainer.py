from metrics.logger import Logger
from utils.utils import DatasetType
from tqdm.auto import tqdm
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
import torch
import torch.optim as optim
from torch.nn import Module
from torch.nn import L1Loss
from config import Config
from models.sotre.wandb_store import WandbStore
from dataset.dataset import ClothesDataset
from models.sotre.model_store import ModelStore
from trainer.common_trainer import LossTracker, VGGPerceptualLoss
from trainer.trainer_configuration import TrainerConfiguration
from trainer.trainer import Trainer

class CGANTrainerConfiguration(TrainerConfiguration):
    def __init__(self, model: Module, discriminator: Module, scheduler: str = None):
        super(CGANTrainerConfiguration, self).__init__("cgan_v1", {"model": model, "scheduler": scheduler, "discriminator": discriminator })


class CGANTrainer(Trainer):
    def __init__(self, trainer_configuration: CGANTrainerConfiguration):
        super(CGANTrainer, self).__init__()
        self.scheduler = None
        self.optimizer = trainer_configuration.optimizer
        self.optimizerD = trainer_configuration.optimizerD
        self.model = trainer_configuration.configuration["model"]
        self.discriminator = trainer_configuration.configuration["discriminator"]
        self.add_scheduler = trainer_configuration.configuration["scheduler"] == "onecyclelr"

    def _combined_criterion(
            self,
            perceptual_loss: torch.nn.Module,
            l1_loss: torch.nn.Module,
            ssim: StructuralSimilarityIndexMeasure,
            c1_weight: float,
            errG,
            outputs,
            target,
    ):
        result = 0
        l1 = 200 * l1_loss(outputs, target)
        result += l1
        
        outputs = ClothesDataset.unnormalize(outputs)
        target = ClothesDataset.unnormalize(target)
        
        # It is important to unnormalize the images before passing them to the perceptual loss
        perceptual = c1_weight * perceptual_loss(outputs, target)
        # result += perceptual

        ssim_res = (ssim.data_range-ssim(outputs, target))
        # result += ssim_res

        result += errG

        return result, l1, perceptual, ssim_res

    def train_model(
        self,
        device,
        train_dataloader,
        val_dataloader,
        cfg: Config,
        logger: Logger,
        remote_model_store: WandbStore,
        local_model_store: ModelStore,
        start_from_epoch: int = 0,
    ):
        optimizer: optim.Optimizer = self.optimizer
        model: Module = self.model
        optimizerD: optim.Optimizer = self.optimizerD
        discriminator: Module = self.discriminator
        num_epochs = cfg.num_epochs
        max_batches = cfg.max_batches
        ssim_range = cfg.ssim_range

        c1_loss = VGGPerceptualLoss().to(device) #None
        c2_loss = L1Loss() #None
        ssim = StructuralSimilarityIndexMeasure(data_range=ssim_range).to(device)

        print("Training started")
        epochs = tqdm(total=num_epochs, desc="Epochs", initial=start_from_epoch)
        training_steps = len(train_dataloader.data_loader)
        validation_steps = len(val_dataloader.data_loader)
        training_progress = tqdm(total=training_steps, desc="Training progress")
        validation_progress = tqdm(total=validation_steps, desc="Validation progress")

        checkpoint_file = ""

        for epoch in range(num_epochs):
            # Fix for tqdm not starting from start_from_epoch
            if epoch < start_from_epoch:
                continue
            training_progress.reset()
            validation_progress.reset()
            model.train()
            loss_tracker = LossTracker(epoch)
            self._forward_step(
                device,
                model,
                train_dataloader.data_loader,
                DatasetType.TRAIN,
                c1_loss,
                c2_loss,
                ssim,
                optimizer,
                training_progress,
                validation_progress,
                discriminator,
                optimizerD,
                loss_tracker,
                max_batches
            )

            # We don't want to disable dropout:
            # model.eval()
            self._forward_step(
                device,
                model,
                val_dataloader.data_loader,
                DatasetType.VALIDATION,
                c1_loss,
                c2_loss,
                ssim,
                optimizer,
                training_progress,
                validation_progress,
                discriminator,
                optimizerD,
                loss_tracker,
                max_batches
            )

            _, _, _, _, _, _, _, _, _, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss_avg:.4f}, '
                f'Validation Loss: {val_loss_avg:.4f}')
            

            checkpoint_file = local_model_store.save_model(cfg=cfg, model=model, optimizer=optimizer, discriminator=discriminator, optimizerD=optimizerD, epoch=epoch, loss=train_loss_avg, val_loss=val_loss_avg)

            logger.log_training(epoch, loss_tracker)
            with torch.no_grad():
                num_images_remote = 16
                train_target = [train_dataloader.data_loader.dataset[i] for i in range(0, num_images_remote)]
                val_target = [val_dataloader.data_loader.dataset[i] for i in range(0, num_images_remote)]
                ten_train = [ClothesDataset.unnormalize(model(img["centered_mask_body"].to(device).unsqueeze(0))) for img in train_target]
                ten_val = [ClothesDataset.unnormalize(model(img["centered_mask_body"].to(device).unsqueeze(0))) for img in val_target]
                train_target = [ClothesDataset.unnormalize(img["target"].to(device).unsqueeze(0)) for img in train_target]
                val_target = [ClothesDataset.unnormalize(img["target"].to(device).unsqueeze(0)) for img in val_target]
                logger.log_images(epoch, ten_train, ten_val, train_target, val_target)

            epochs.update()

        if cfg.wandb_save_checkpoint:
            remote_model_store.save_model(checkpoint_file)


        print("Training completed!")
        return model

    def _forward_step(
            self,
            device,
            model: torch.nn.Module,
            loader: torch.utils.data.DataLoader,
            dataset_type: DatasetType,
            c1Loss: torch.nn.Module,
            c2Loss: torch.nn.Module,
            ssim: StructuralSimilarityIndexMeasure,
            optimizer: torch.optim.Optimizer,
            training_progress: tqdm,
            validation_progress: tqdm,
            discriminator: torch.nn.Module,
            optimizerD: torch.optim.Optimizer,
            loss_tracker: LossTracker,
            max_batches: int = 0,
    ):
        perceptual_weight = 0.3

        Dcriterion = torch.nn.BCELoss()

        for batch_idx, inputs in enumerate(loader):
            target = inputs["target"].to(device)
            source = inputs["centered_mask_body"].to(device)
            if 0 < max_batches and max_batches == batch_idx:
                break
            if dataset_type == DatasetType.TRAIN:
                discriminator.zero_grad()
                output = discriminator(source, target).squeeze()
                ones = torch.ones(output.shape, dtype=torch.float, device=device)
                errD = Dcriterion(output, ones)
                _, pred = model(source)
                pred = pred.detach()
                print('Shape:', pred.shape)
                zeros = torch.zeros(output.shape, dtype=torch.float, device=device)
                output = discriminator(source, pred).squeeze()
                errD_fake = Dcriterion(output, zeros)
                errD = (errD + errD_fake) * 0.5
                errD.backward()

                optimizerD.step()
                model.zero_grad()
                _, pred = model(source)
                ones = torch.ones(output.shape, dtype=torch.float, device=device)
                output = discriminator(source, pred).squeeze()
                errG = Dcriterion(output, ones)
                loss, l1, perceptual, ssim_res = self._combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
                loss.backward()
                optimizer.step()
                training_progress.update()
                
                loss_tracker.train_l1.append(l1.item())
                loss_tracker.train_perceptual.append(perceptual.item())
                loss_tracker.train_ssim.append(ssim_res.item())
                loss_tracker.train_generator_loss.append(errG.item())
                loss_tracker.train_discriminator_loss.append(errD.item())
                loss_tracker.train_loss.append(loss.item())
                
            else:
                with torch.no_grad():
                    _, pred = model(source)
                    output = discriminator(source, pred).squeeze()
                    ones = torch.ones(output.shape, dtype=torch.float, device=device)
                    errG = Dcriterion(output, ones)
                    loss, l1, perceptual, ssim_res = self._combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
                
                loss_tracker.val_l1.append(l1.item())
                loss_tracker.val_perceptual.append(perceptual.item())
                loss_tracker.val_ssim.append(ssim_res.item())
                loss_tracker.val_generator_loss.append(errG.item())
                loss_tracker.val_loss.append(loss.item())
                validation_progress.update()
