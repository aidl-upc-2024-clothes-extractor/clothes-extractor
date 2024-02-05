import torch
import torch.optim as optim
from torch.nn import L1Loss
from utils import utils
from tqdm.auto import tqdm

import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
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

def combined_criterion(c1, c2, w, outputs, target):
    return w * c1(outputs, target) + c2(outputs, target)
    
def train_model(model, device, train_dataloader, val_dataloader, num_epochs, learning_rate, max_batches=0):
    c1 = VGGPerceptualLoss().to(device)
    c2 = L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    w = 0.3

    print('Start training')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, inputs in enumerate(tqdm(train_dataloader.data_loader)):
            target = inputs ["target"].to(device)
            source = inputs["centered_mask_body"].to(device)
            optimizer.zero_grad()

            outputs = model(source)
            loss = combined_criterion(c1, c2, w, outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if 0 < max_batches == batch_idx :
                break


        avg_train_loss = running_loss / len(train_dataloader.data_loader)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(val_dataloader.data_loader):
                target = inputs["target"].to(device)
                source = inputs["centered_mask_body"].to(device)

                outputs = model(source)
                loss = combined_criterion(c1, c2, w, outputs, target)

                running_loss += loss.item()

                if 0 < max_batches == batch_idx:
                    break

        avg_val_loss = running_loss / len(val_dataloader.data_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}')

    print('Finished Training')
    return model