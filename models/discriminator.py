from torch import nn
import torch

NDF = 64

class DiscriminatorReduction(nn.Module):
    def __init__(self):
        super(DiscriminatorReduction, self).__init__()
        nc = 3
        ndf = NDF
        self.main = nn.Sequential(
            # input is ``(nc) x 224 x 224``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ``(ndf) x 112 x 112``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf * 2) x 56 x 56``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 28 x 28``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 14 x 14``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*16) x 7 x 7``
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = DiscriminatorReduction()

    def resize(self, im):
        return nn.functional.interpolate(im, size=(224, 224), mode='bilinear', align_corners=True)
    
    def forward(self, i1, i2, source):
        i1 = self.resize(i1)
        i2 = self.resize(i2)
        source = self.resize(source)

        concated = torch.cat((i1, i2, source), 1)
        
        return torch.flatten(self.main(concated))
        
        
