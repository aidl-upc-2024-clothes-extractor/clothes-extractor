from torch import nn
import torch

NDF = 64

class DiscriminatorReduction(nn.Module):
    def __init__(self):
        super(DiscriminatorReduction, self).__init__()
        nc = 3
        ndf = NDF
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main1 = DiscriminatorReduction()
        self.main2 = DiscriminatorReduction()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(NDF * 4 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def resize(self, im):
        return nn.functional.interpolate(im, size=(128, 128), mode='bilinear', align_corners=True)
    
    def forward(self, i1, i2):
        i1 = self.resize(i1)
        i2 = self.resize(i2)

        i1 = self.main1(i1)
        i2 = self.main2(i2)

        i1 = self.flatten(i1)
        i2 = self.flatten(i2)

        concated = torch.cat((i1, i2), 1)
        
        return self.fc1(concated)
        
        
