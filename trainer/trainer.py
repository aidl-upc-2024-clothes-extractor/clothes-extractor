import numpy as np
import torch
import torch.optim as optim
from torch.nn import Module
from torch.nn import L1Loss
from config import Config
from models.wandb_store import WandbStore
from metrics.logger import Logger
from utils.utils import DatasetType
from tqdm.auto import tqdm
import math
from dataset.dataset import ClothesDataset, ClothesDataLoader

import torchvision
from models.model_store import ModelStore
from torchmetrics.image import StructuralSimilarityIndexMeasure

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        #blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def combined_criterion(
        perceptual_loss: torch.nn.Module,
        l1_loss: torch.nn.Module,
        ssim: StructuralSimilarityIndexMeasure,
        c1_weight: float,
        errG,
        outputs,
        target,
):
    result = 0
    perceptual = 0
    ssim_res = 0
    if l1_loss is not None:
        result += 100 * l1_loss(outputs, target)
    outputs = outputs / 2 + 0.5
    target = target / 2 + 0.5
    if perceptual_loss is not None:
        # It is important to unnormalize the images before passing them to the perceptual loss
        perceptual = c1_weight * perceptual_loss(outputs, target)
        # result += perceptual
    if ssim is not None:
        ssim_res = (ssim.data_range-ssim(outputs, target))
        # result += ssim_res
    if errG is not None:
        result += errG
    return result, perceptual, ssim_res



def train_model(
    optimizer: optim.Optimizer,
    model: Module,
    optimizerD: optim.Optimizer,
    discriminator: Module,
    device,
    train_dataloader,
    val_dataloader,
    cfg: Config,
    logger: Logger,
    remote_model_store: WandbStore,
    local_model_store: ModelStore,
    start_from_epoch: int = 0,
):
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

    for epoch in range(num_epochs):
        # Fix for tqdm not starting from start_from_epoch
        if epoch < start_from_epoch:
            continue
        training_progress.reset()
        validation_progress.reset()
        model.train()
        train_loss, _, _, train_generator_loss, discriminator_loss = forward_step(
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
            max_batches
        )
        train_loss_avg = np.mean(train_loss)
        train_generator_loss_avg = np.mean(train_generator_loss)
        train_discriminator_loss_avg = np.mean(discriminator_loss)


        model.eval()
        val_loss, percetual_loss, ssim_loss, generator_loss, _ = forward_step(
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
            max_batches
        )
        val_loss_avg = np.mean(val_loss)
        percetual_loss_avg = np.mean(percetual_loss)
        ssim_loss_avg = np.mean(ssim_loss)
        eval_generator_loss_avg = np.mean(generator_loss)


        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss_avg:.4f}, '
              f'Validation Loss: {val_loss_avg:.4f}')
        

        if (epoch+1) % cfg.checkpoint_save_frequency == 0 or epoch+1 == num_epochs:
            checkpoint_file = local_model_store.save_model(cfg=cfg, model=model, optimizer=optimizer, discriminator=discriminator, optimizerD=optimizerD, epoch=epoch, loss=train_loss_avg, val_loss=val_loss_avg)
            if cfg.wandb_save_checkpoint:
                remote_model_store.save_model(checkpoint_file)

        logger.log_training(epoch, train_loss_avg, val_loss_avg, percetual_loss_avg, ssim_loss_avg, train_generator_loss_avg, eval_generator_loss_avg, train_discriminator_loss_avg)
        with torch.no_grad():
            ten_train = [model(train_dataloader.data_loader.dataset[i]["centered_mask_body"].to(device).unsqueeze(0)) for i in range(0, 16)]
            ten_train = [ClothesDataset.unnormalize(x) for x in ten_train]
            ten_val = [model(val_dataloader.data_loader.dataset[i]["centered_mask_body"].to(device).unsqueeze(0)) for i in range(0, 16)]
            ten_val = [ClothesDataset.unnormalize(x) for x in ten_val]
            logger.log_images(epoch, ten_train, ten_val)

        epochs.update()


    print("Training completed!")
    return model

def forward_step(
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
        max_batches: int = 0,
):
    perceptual_weight = 0.3
    loss_list = []
    perceptual_list = []
    ssim_list = []
    generator_loss = []
    discriminator_loss = []

    Dcriterion = torch.nn.BCELoss()

    for batch_idx, inputs in enumerate(loader):
        target = inputs["target"].to(device)
        source = inputs["centered_mask_body"].to(device)
        if dataset_type == DatasetType.TRAIN:
            discriminator.zero_grad()
            output = discriminator(source, target).squeeze()
            ones = torch.ones(output.shape, dtype=torch.float, device=device)
            errD = Dcriterion(output, ones)
            pred = model(source).detach()
            zeros = torch.zeros(output.shape, dtype=torch.float, device=device)
            output = discriminator(source, pred).squeeze()
            errD_fake = Dcriterion(output, zeros)
            errD = (errD + errD_fake) * 0.5
            errD.backward()

            optimizerD.step()
            model.zero_grad()
            pred = model(source)
            ones = torch.ones(output.shape, dtype=torch.float, device=device)
            output = discriminator(source, pred).squeeze()
            errG = Dcriterion(output, ones)
            loss, perceptual, ssim_res = combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
            loss.backward()
            optimizer.step()
            training_progress.update()
            discriminator_loss.append(errD.item())
        else:
            with torch.no_grad():
                pred = model(source)
                ones = torch.ones(output.shape, dtype=torch.float, device=device)
                output = discriminator(source, pred).squeeze()
                errG = Dcriterion(output, ones)
                loss, perceptual, ssim_res = combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
            validation_progress.update()
        loss_list.append(loss.item())
        perceptual_list.append(perceptual.item())
        ssim_list.append(ssim_res.item())
        generator_loss.append(errG.item())

    return loss_list, perceptual_list, ssim_list, generator_loss, discriminator_loss
