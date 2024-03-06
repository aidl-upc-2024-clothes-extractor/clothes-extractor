from metrics.logger import Logger
from pix2pix.models.pix2pix_model import Pix2PixModel
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

class Pix2PixTrainerConfiguration(TrainerConfiguration):
    def __init__(self, model: Module):
        super(Pix2PixTrainerConfiguration, self).__init__("cgan_v1", {"model": model})


class Pix2PixTrainer(Trainer):
    def __init__(self, trainer_configuration: Pix2PixTrainerConfiguration):
        super(Pix2PixTrainer, self).__init__()
        self.model = trainer_configuration.configuration["model"]

    def _combined_criterion(
            self,
            perceptual_loss: torch.nn.Module,
            l1_loss: torch.nn.Module,
            ssim: StructuralSimilarityIndexMeasure,
            c1_weight: float,
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
        model: Pix2PixModel = self.model
        num_epochs = cfg.num_epochs
        max_batches = cfg.max_batches
        ssim_range = cfg.ssim_range

        c1_loss = VGGPerceptualLoss().to(device) #None
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
            model.update_learning_rate()

            training_progress.reset()
            validation_progress.reset()
            # model.train()
            loss_tracker = LossTracker(epoch)
            self._forward_step(
                device,
                model,
                train_dataloader.data_loader,
                DatasetType.TRAIN,
                c1_loss,
                ssim,
                None,
                training_progress,
                validation_progress,
                None,
                None,
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
                ssim,
                None,
                training_progress,
                validation_progress,
                None,
                None,
                loss_tracker,
                max_batches
            )

            _, _, _, _, _, _, _, _, _, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss_avg:.4f}, '
                f'Validation Loss: {val_loss_avg:.4f}')
            

            checkpoint_file = local_model_store.save_model(cfg=cfg, model=model, optimizer=None, discriminator=None, optimizerD=None, epoch=epoch, loss=train_loss_avg, val_loss=val_loss_avg)

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
            model: Pix2PixModel,
            loader: torch.utils.data.DataLoader,
            dataset_type: DatasetType,
            c1Loss: torch.nn.Module,
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
                model.set_input({
                    'A': source,
                    'B': target,
                    'A_paths': 'RANDOM_STRING'
                })
                model.optimize_parameters()
                pred = model.fake_B
                training_progress.update()
                
                loss_tracker.train_l1.append(model.loss_G_L1.item())
                _, _, perceptual, ssim_res = self._combined_criterion(c1Loss, ssim, perceptual_weight, pred, target)
                loss_tracker.train_perceptual.append(perceptual.item())
                loss_tracker.train_ssim.append(ssim_res.item())
                loss_tracker.train_generator_loss.append(model.loss_G_GAN.item())
                loss_tracker.train_discriminator_loss.append(model.loss_D.item())
                loss_tracker.train_loss.append(model.loss_G.item())
                
            else:
                with torch.no_grad():
                    pred = model(source)
                    _, l1, perceptual, ssim_res = self._combined_criterion(c1Loss, ssim, perceptual_weight, pred, target)
                    pred_fake = model.netD(pred)
                    loss_G = model.criterionGAN(pred_fake, True)
                loss_tracker.val_l1.append(l1.item())
                loss_tracker.val_perceptual.append(perceptual.item())
                loss_tracker.val_ssim.append(ssim_res.item())
                loss_tracker.val_generator_loss.append(loss_G.item())
                loss_tracker.val_loss.append((loss_G + l1).item())
                validation_progress.update()
