import numpy as np
import torch
import torch.optim as optim
from torch.nn import L1Loss
from config import Config
from models.wandb_store import WandbStorer
from metrics.logger import Logger
from utils.utils import DatasetType

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
        c1_loss: torch.nn.Module,
        c2_loss: torch.nn.Module,
        ssim: StructuralSimilarityIndexMeasure,
        w: float,
        outputs,
        target,
):
    result = 0
    if c2_loss is not None:
        result += c2_loss(outputs, target)
    if c1_loss is not None:
        result += w * c1_loss(outputs, target)
    if ssim is not None:
        result += (ssim.data_range-ssim(outputs, target))
    return result


def train_model(
        model,
        device,
        train_dataloader,
        val_dataloader,
        cfg: Config,
        logger: Logger,
        model_storer: WandbStorer,
):
    num_epochs = cfg.num_epochs
    learning_rate = cfg.learning_rate
    max_batches = cfg.max_batches
    reload_model = cfg.reload_model
    ssim_range = cfg.ssim_range

    c1_loss = VGGPerceptualLoss().to(device) #None
    c2_loss = L1Loss() #None
    ssim = StructuralSimilarityIndexMeasure(data_range=ssim_range).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    m_storer = ModelStore()

    print(reload_model)
    if reload_model is not None and reload_model != "None":
        model, optimizer, epoch, loss = m_storer.load_model(model=model, optimizer=optimizer, model_name=reload_model)

    if reload_model is not None and reload_model == "latest":
        reload_model = None
        model, optimizer, epoch, loss = m_storer.load_model(model=model, optimizer=optimizer, model_name=reload_model)


    print('Start training')
    for epoch in range(num_epochs):
        model.train()
        train_loss = forward_step(
            device,
            model,
            train_dataloader.data_loader,
            DatasetType.TRAIN,
            c1_loss,
            c2_loss,
            ssim,
            optimizer,
        )
        train_loss_avg = np.mean(train_loss)

        model.eval()
        val_loss = forward_step(
            device,
            model,
            val_dataloader.data_loader,
            DatasetType.VALIDATION,
            c1_loss,
            c2_loss,
            ssim,
            optimizer,
        )
        val_loss_avg = np.mean(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss_avg:.4f}, '
              f'Validation Loss: {val_loss_avg:.4f}')

        if (epoch+1) % 2 == 0 or epoch+1 == num_epochs:
            checkpoint_file = m_storer.save_model(model=model, optimizer=optimizer, epoch=epoch, loss=train_loss_avg)
            model_storer.save_model(checkpoint_file)

        logger.log_training(epoch, train_loss_avg, val_loss_avg)

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
):
    w = 0.3
    loss_list = []

    for batch_idx, inputs in enumerate(loader):
        if dataset_type == DatasetType.TRAIN:
            target = inputs["target"].to(device)
            source = inputs["centered_mask_body"].to(device)
            optimizer.zero_grad()

            outputs = model(source)
            loss = combined_criterion(c1Loss, c2Loss, ssim, w, outputs, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                target = inputs["target"].to(device)
                source = inputs["centered_mask_body"].to(device)
                outputs = model(source)
                loss = combined_criterion(c1Loss, c2Loss, ssim, w, outputs, target)
        loss_list.append(loss.item())

    return loss_list
