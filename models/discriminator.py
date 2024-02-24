from torch import nn
import torch

NDF = 64

class DiscriminatorReduction(nn.Module):
    def __init__(self):
        super(DiscriminatorReduction, self).__init__()
        nc = 6
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
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``1 x 7 x 7``
            nn.Conv2d(ndf * 8, 1, 7, 1, 0, bias=False),
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
    
    def forward(self, i1, source):
        i1 = self.resize(i1)
        source = self.resize(source)

        concated = torch.cat((i1, source), 1)
        
        return self.main(concated)
        
        
