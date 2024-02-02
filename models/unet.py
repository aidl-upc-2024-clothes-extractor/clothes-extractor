import torch
import torch.nn.functional as F
from torch import nn


class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.activate_attention = False

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        if self.activate_attention:
            self.attn1 = SelfAttention(n_feat)

        self.down2 = UnetDown(n_feat, 2 * n_feat)
        if self.activate_attention:
            self.attn2 = SelfAttention(2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            #nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        if self.activate_attention:
            self.attn1up = SelfAttention(n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        if self.activate_attention:
            self.attn2up = SelfAttention(n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x):
        # Downsampling
        debug = False
        if debug:
            print(f'Entry: {x.shape}')
        x = self.init_conv(x)
        if debug:
            print(f'1: {x.shape}')
        if self.activate_attention:
            down1 = self.attn1(self.down1(x))
        down1 = self.down1(x)
        if debug:
            print(f'down1: {down1.shape}')
        if self.activate_attention:
            down2 = self.attn2(self.down2(down1))
        down2 = self.down2(down1)
        if debug:
            print(f'down2: {down2.shape}')

        hiddenvec = self.to_vec(down2)
        if debug:
            print(f'hiddenvec: {hiddenvec.shape}')

        # Upsampling
        up1 = self.up0(hiddenvec)
        if debug:
            print(f'up1: {up1.shape}')

        condition = up1
        if debug:
            print(f'condition: {condition.shape}')
            print('S', condition.shape, down2.shape)
        if self.activate_attention:
            up2 = self.attn1up(self.up1(condition, down2))
        up2 = self.up1(condition, down2)
        if debug:
            print(f'up2: {up2.shape}')

        condition = up2
        if debug:
            print(f'condition: {condition.shape}')

        if self.activate_attention:
            up3 = self.attn2up(self.up2(condition, down1))
        up3 = self.up2(condition, down1)
        if debug:
            print(f'up3: {up3.shape}')
        out = self.out(torch.cat((up3, x), 1))
        if debug:
            exit(0)
        return out


class SelfAttention(nn.Module):
    """
    Similar to the Transformer Architecture, this network has self-attention blocks.
    """

    def __init__(self, n_channels):
        super().__init__()
        n_channels_out = n_channels // 4
        self.query = nn.Linear(n_channels, n_channels_out, bias=False)
        self.key = nn.Linear(n_channels, n_channels_out, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).view(B, H * W, C)  # Shape: [B, H*W, C]

        q = self.query(x)  # [B, H*W, C]
        k = self.key(x)  # [B, H*W, C]
        v = self.value(x)  # [B, H*W, C]

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=1)  # Shape: [B, H*W, H*W]
        out = self.gamma * torch.bmm(attn, v) + x  # Shape: [B, H*W, C]

        out = out.permute(0, 2, 1).view(B, C, H, W).contiguous()

        return out


class ResidualConvBlock(nn.Module):
    """
    The following are resnet block, which consist of convolutional layers,
    followed by batch normalization and residual connections.
    """

    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        """
        process and downscale the image feature maps
        """
        layers = [
            ResidualConvBlock(in_channels, in_channels, True),
            ResidualConvBlock(in_channels, in_channels, True),
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        """
        process and upscale the image feature maps
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels, True),
            ResidualConvBlock(out_channels, out_channels, True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
