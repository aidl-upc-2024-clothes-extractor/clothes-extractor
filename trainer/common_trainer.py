
import torchvision
import numpy as np
import torch

class LossTracker:
    def __init__(self, epoch: int):
        self.epoch = epoch
        self.train_l1 = []
        self.val_l1 = []
        self.train_perceptual = []
        self.val_perceptual = []
        self.train_ssim = []
        self.val_ssim = []
        self.train_generator_loss = []
        self.val_generator_loss = []
        self.train_discriminator_loss = []
        self.train_loss = []
        self.val_loss = []

    def get_avgs(self):
        train_l1_avg = np.mean(self.train_l1)
        val_l1_avg = np.mean(self.val_l1)
        train_perceptual_avg = np.mean(self.train_perceptual)
        val_perceptual_avg = np.mean(self.val_perceptual)
        train_ssim_avg = np.mean(self.train_ssim)
        val_ssim_avg = np.mean(self.val_ssim)
        if len(self.train_generator_loss) == 0:
            train_generator_loss_avg = 0
            val_generator_loss_avg = 0
        else:
            train_generator_loss_avg = np.mean(self.train_generator_loss)
            val_generator_loss_avg = np.mean(self.val_generator_loss)
        if len(self.train_discriminator_loss) == 0:
            train_discriminator_loss_avg = 0
        else:
            train_discriminator_loss_avg = np.mean(self.train_discriminator_loss)

        train_loss_avg = np.mean(self.train_loss)
        val_loss_avg = np.mean(self.val_loss)

        return train_l1_avg, val_l1_avg, train_perceptual_avg, val_perceptual_avg, train_ssim_avg, val_ssim_avg, train_generator_loss_avg, val_generator_loss_avg, train_discriminator_loss_avg, train_loss_avg, val_loss_avg


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
                .features[:4]
                .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
                .features[4:9]
                .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
                .features[9:16]
                .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
                .features[16:23]
                .eval()
        )
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
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