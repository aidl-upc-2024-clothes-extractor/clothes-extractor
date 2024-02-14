import numpy as np
import torch
import torch.optim as optim
from torch.nn import L1Loss
from config import Config
from models.wandb_store import WandbStorer
from metrics.logger import Logger
from utils.utils import DatasetType
from tqdm import tqdm
import math
from model_instantiate import get_model

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
        result += l1_loss(outputs, target)
    if perceptual_loss is not None:
        perceptual = c1_weight * perceptual_loss(outputs, target)
        # result += perceptual
    if ssim is not None:
        ssim_res = (ssim.data_range-ssim(outputs, target))
        # result += ssim_res
    if errG is not None:
        result += errG
    return result, perceptual, ssim_res


def train_model(
        device,
        train_dataloader,
        val_dataloader,
        wandb_run,
        cfg: Config,
        logger: Logger,
        model_storer: WandbStorer,
):
    num_epochs = cfg.num_epochs
    max_batches = cfg.max_batches
    ssim_range = cfg.ssim_range

    c1_loss = VGGPerceptualLoss().to(device) #None
    c2_loss = L1Loss() #None
    ssim = StructuralSimilarityIndexMeasure(data_range=ssim_range).to(device)
    
    model, optimizerG, discriminator, optimizerD, epoch, loss = get_model(cfg, device)
    # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch
    wandb_run.watch(models=model) #, log=UtLiteral["gradients", "weights"])

    local_storer = ModelStore()

    print('Start training')
    epochs = tqdm(range(num_epochs), desc="Epochs")
    training_steps = len(train_dataloader.data_loader)
    validation_steps = len(val_dataloader.data_loader)
    training_progress = tqdm(total=training_steps, desc="Training progress")
    validation_progress = tqdm(total=validation_steps, desc="Validation progress")

    for epoch in epochs:
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
            optimizerG,
            training_progress,
            validation_progress,
            discriminator,
            optimizerD
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
            optimizerG,
            training_progress,
            validation_progress,
            discriminator,
            optimizerD
        )
        val_loss_avg = np.mean(val_loss)
        percetual_loss_avg = np.mean(percetual_loss)
        ssim_loss_avg = np.mean(ssim_loss)
        eval_generator_loss_avg = np.mean(generator_loss)

        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss_avg:.4f}, '
              f'Validation Loss: {val_loss_avg:.4f}')

        if (epoch+1) % 2 == 0 or epoch+1 == num_epochs:
            checkpoint_file = local_storer.save_model(model=model, optimizer=optimizerG, discriminator=discriminator, optimizerD=optimizerD, epoch=epoch, loss=train_loss_avg)
            model_storer.save_model(checkpoint_file)

        logger.log_training(epoch, train_loss_avg, val_loss_avg, percetual_loss_avg, ssim_loss_avg, train_generator_loss_avg, eval_generator_loss_avg, train_discriminator_loss_avg)

    print('Finished Training')
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
        optimizerD: torch.optim.Optimizer
):
    perceptual_weight = 0.3
    loss_list = []
    perceptual_list = []
    ssim_list = []
    generator_loss = []
    discriminator_loss = []
    real_first_label = 0.
    real_second_label = 1.
    optimum_label = 0.5


    Dcriterion = torch.nn.BCELoss()

    for batch_idx, inputs in enumerate(loader):
        target = inputs["target"].to(device)
        source = inputs["centered_mask_body"].to(device)
        if dataset_type == DatasetType.TRAIN:
            discriminator.zero_grad()
            # Format batch
            b_size = target.size(0)
            pred = model(source)
            i1 = target
            i2 = pred.detach()
            random_side = np.random.randint(0, 2)
            label = torch.full((b_size,), real_first_label, dtype=torch.float, device=device)
            if random_side != 0:
                i1, i2 = i2, i1
                label.fill_(real_second_label)
            # Forward pass real batch through D
            output = discriminator(i1, i2, source).view(-1)
            # Calculate loss on all-real batch
            errD = Dcriterion(output, label)
            # Calculate gradients for D in backward pass
            errD.backward()
            D_x = output.mean().item()

            optimizerD.step()
            model.zero_grad()
            label.fill_(optimum_label)
            i1 = target
            i2 = pred
            random_side = np.random.randint(0, 2)
            if random_side != 0:
                i1, i2 = i2, i1
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(i1, i2, source).view(-1)
            # Calculate G's loss based on this output
            errG = Dcriterion(output, label)
            # Calculate gradients for G
            D_G_z2 = output.mean().item()
            # Update G
            loss, perceptual, ssim_res = combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
            loss.backward()
            optimizer.step()
            training_progress.update()
            discriminator_loss.append(errD.item())
        else:
            with torch.no_grad():
                pred = model(source)
                i1 = target
                i2 = pred
                random_side = np.random.randint(0, 2)
                if random_side != 0:
                    i1, i2 = i2, i1
                output = discriminator(i1, i2, source).view(-1)
                label = torch.full((output.size(0),), optimum_label, dtype=torch.float, device=device)
                errG = Dcriterion(output, label)
                loss, perceptual, ssim_res = combined_criterion(c1Loss, c2Loss, ssim, perceptual_weight, errG, pred, target)
            validation_progress.update()
        loss_list.append(loss.item())
        perceptual_list.append(perceptual.item())
        ssim_list.append(ssim_res.item())
        generator_loss.append(errG.item())

    return loss_list, perceptual_list, ssim_list, generator_loss, discriminator_loss
