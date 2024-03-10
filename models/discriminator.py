from torch import nn
import torch
import numpy as np

NDF = 16

class DiscriminatorReduction(nn.Module):
    def __init__(self):
        super(DiscriminatorReduction, self).__init__()
        nc = 9
        ndf = NDF
        self.main = nn.Sequential(
            # input is ``(nc) x 224 x 224``
            nn.Conv2d(nc, ndf, 1, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ``224 x 224``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``112 x 112``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``56 x 56``
            nn.Conv2d(ndf * 4, 1, 1, 1, 1),
            # nn.BatchNorm2d(ndf * 5),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. ``7 x 7``
            # nn.Conv2d(ndf * 5, 1, 7, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = DiscriminatorReduction()

    def resize(self, im):
        if im.shape[2] == 224 and im.shape[3] == 224:
            return im
        return nn.functional.interpolate(im, size=(224, 224), mode='bilinear', align_corners=True)
    
    def forward(self, input, source):
        input = self.resize(input)
        source = self.resize(source)

        concated = torch.cat((input, source), 1)
        
        return self.main(concated)
        
        

class ConditionalFCCGANDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalFCCGANDiscriminator, self).__init__()
        # Define the initial number of channels after the input layer
        initial_channels = 16  # Base number of channels for the first convolutional layer

        self.conv_layers = nn.Sequential(
            # Adjust the first layer to take 6-channel input
            nn.Conv2d(6, initial_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Increment the channels by a fixed multiplier in each layer
            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(initial_channels * 4, initial_channels * 8, kernel_size=3, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(initial_channels * 8, initial_channels * 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.size_after_conv = self._get_conv_output_size((1, 6, 224, 224))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.size_after_conv, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():  # Ensuring this operation doesn't track gradients
            input = torch.rand(*shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))

    def resize(self, im):
        if im.shape[2] == 224 and im.shape[3] == 224:
            return im
        return nn.functional.interpolate(im, size=(224, 224), mode='bilinear', align_corners=True)
    
    def forward(self, x1, x2):
        x1 = self.resize(x1)
        x2 = self.resize(x2)
        # Concatenate x1 and x2 along the channel dimension before passing through the network
        x = torch.cat((x1, x2), dim=1)  # Assuming x1 and x2 are [batch_size, 3, 224, 224]
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x