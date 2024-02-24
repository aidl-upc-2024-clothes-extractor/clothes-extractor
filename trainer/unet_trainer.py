import numpy as np
import torch
import torch.optim as optim
from torch.nn import Module
from torch.nn import L1Loss
from config import Config
from models.sotre.wandb_store import WandbStore
from metrics.logger import Logger
from trainer.trainer import Trainer
from trainer.trainer_configuration import TrainerConfiguration
from utils.utils import DatasetType
from tqdm.auto import tqdm
from dataset.dataset import ClothesDataset


from models.sotre.model_store import ModelStore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from trainer.common_trainer import LossTracker, VGGPerceptualLoss



class UnetTrainerConfiguration(TrainerConfiguration):
    def __init__(self, model: Module, scheduler: str = None):
        super(UnetTrainerConfiguration, self).__init__("unet_v1", {"model": model, "scheduler": scheduler})


class UnetTrainer(Trainer):
    def __init__(self, trainer_configuration: UnetTrainerConfiguration):
        super(UnetTrainer, self).__init__()
        self.scheduler = None
        self.optimizer = trainer_configuration.optimizer
        self.model = trainer_configuration.configuration["model"]
        self.add_scheduler = trainer_configuration.configuration["scheduler"] == "onecyclelr"


    def _combined_criterion(
            self,
            perceptual_loss: torch.nn.Module,
            l1_loss: torch.nn.Module,
            ssim: StructuralSimilarityIndexMeasure,
            c1_weight: float,
            outputs,
            target,
    ):
        l1 = l1_loss(outputs, target)
        result = l1

        outputs = ClothesDataset.unnormalize(outputs)
        target = ClothesDataset.unnormalize(target)

        # It is important to unnormalize the images before passing them to the perceptual loss
        perceptual = c1_weight * perceptual_loss(outputs, target)
        result += perceptual

        ssim_res = ssim.data_range - ssim(outputs, target)
        result += ssim_res
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
    ) -> Module:
        num_epochs = cfg.num_epochs
        max_batches = cfg.max_batches
        ssim_range = cfg.ssim_range

        c1_loss = VGGPerceptualLoss().to(device)
        c2_loss = L1Loss()
        ssim = StructuralSimilarityIndexMeasure(data_range=ssim_range).to(device)

        epochs = tqdm(total=num_epochs, desc="Epochs", initial=start_from_epoch)
        training_steps = len(train_dataloader.data_loader)
        validation_steps = len(val_dataloader.data_loader)
        training_progress = tqdm(total=training_steps, desc="Training progress")
        validation_progress = tqdm(total=validation_steps, desc="Validation progress")

        if self.add_scheduler:
            max_lr = cfg.learning_rate / 0.06
            steps_per_epoch = training_steps
            print(f"Using OneCycleLR: max_lr={max_lr} steps_per_epoch={steps_per_epoch} ")
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=cfg.num_epochs
            )

        print("Training started")
        for epoch in range(num_epochs):
            # Fix for tqdm not starting from start_from_epoch
            if epoch < start_from_epoch:
                continue
            training_progress.reset()
            validation_progress.reset()
            self.model.train()
            loss_tracker = LossTracker(epoch)
            self._forward_step(
                device,
                self.model,
                train_dataloader.data_loader,
                DatasetType.TRAIN,
                c1_loss,
                c2_loss,
                ssim,
                self.optimizer,
                training_progress,
                validation_progress,
                loss_tracker,
                max_batches=max_batches,
            )

            self.model.eval()
            self._forward_step(
                device,
                self.model,
                val_dataloader.data_loader,
                DatasetType.VALIDATION,
                c1_loss,
                c2_loss,
                ssim,
                self.optimizer,
                training_progress,
                validation_progress,
                loss_tracker,
                max_batches=max_batches,
            )
            _, _, _, _, _, _, _, _, _, train_loss_avg, val_loss_avg = loss_tracker.get_avgs()

            tqdm.write(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss_avg:.4f}, "
                f"Validation Loss: {val_loss_avg:.4f}"
            )

            checkpoint_file = local_model_store.save_model(
                cfg, self.model, self.optimizer, epoch, train_loss_avg, val_loss_avg
            )

            logger.log_training(epoch, loss_tracker)
            
            with torch.no_grad():
                num_images_remote = 16
                ten_train = [self.model(train_dataloader.data_loader.dataset[i]["centered_mask_body"].to(device).unsqueeze(0)) for i in range(0, num_images_remote)]
                ten_train = [ClothesDataset.unnormalize(x) for x in ten_train]
                ten_val = [self.model(val_dataloader.data_loader.dataset[i]["centered_mask_body"].to(device).unsqueeze(0)) for i in range(0, num_images_remote)]
                ten_val = [ClothesDataset.unnormalize(x) for x in ten_val]
                train_target = [ClothesDataset.unnormalize(train_dataloader.data_loader.dataset[i]["target"].to(device).unsqueeze(0)) for i in range(0, num_images_remote)]
                val_target = [ClothesDataset.unnormalize(val_dataloader.data_loader.dataset[i]["target"].to(device).unsqueeze(0)) for i in range(0, num_images_remote)]
                logger.log_images(epoch, ten_train, ten_val, train_target, val_target)

            epochs.update()

            if cfg.wandb_save_checkpoint:
                if len(local_model_store.models_saved) > 0:
                    remote_model_store.save_model(local_model_store.models_saved[-1][0])
                else:
                    remote_model_store.save_model(checkpoint_file)

        print("Training completed!")
        return self.model

    def _forward_step(
            self,
            device,
            model: torch.nn.Module,
            loader: torch.utils.data.DataLoader,
            dataset_type: DatasetType,
            c1_loss: torch.nn.Module,
            c2_loss: torch.nn.Module,
            ssim: StructuralSimilarityIndexMeasure,
            optimizer: torch.optim.Optimizer,
            training_progress: tqdm,
            validation_progress: tqdm,
            loss_tracker: LossTracker,
            max_batches: int = 0,
    ):
        perceptual_weight = 0.3

        for batch_idx, inputs in enumerate(loader):
            if 0 < max_batches and max_batches == batch_idx:
                break
            if dataset_type == DatasetType.TRAIN:
                target = inputs["target"].to(device)
                source = inputs["centered_mask_body"].to(device)
                optimizer.zero_grad()
                outputs = model(source)
                loss, l1, perceptual, ssim_res = self._combined_criterion(
                    c1_loss, c2_loss, ssim, perceptual_weight, outputs, target
                )
                loss.backward()
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                training_progress.update()
                loss_tracker.train_l1.append(l1.item())
                loss_tracker.train_perceptual.append(perceptual.item())
                loss_tracker.train_ssim.append(ssim_res.item())
                loss_tracker.train_loss.append(loss.item())
            else:
                with torch.no_grad():
                    target = inputs["target"].to(device)
                    source = inputs["centered_mask_body"].to(device)
                    outputs = model(source)
                    loss, l1, perceptual, ssim_res = self._combined_criterion(
                        c1_loss, c2_loss, ssim, perceptual_weight, outputs, target
                    )
                loss_tracker.val_l1.append(l1.item())
                loss_tracker.val_perceptual.append(perceptual.item())
                loss_tracker.val_ssim.append(ssim_res.item())
                loss_tracker.val_loss.append(loss.item())
                validation_progress.update()
