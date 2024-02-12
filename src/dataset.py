from torch.utils import data
from os import path
import numpy as np
from torchvision import transforms, io
from torchvision.transforms import functional as F
from PIL import Image
import torch
import os
from src.data_augmentation import RGBAtoRGBWhiteBlack, MakeSquareWithPad, ToFloatTensor, CropAlphaChannelTransform, AlphaToBlackTransform, PadToShapeTransform, SingleChannelToRGBTransform, ApplyMaskTransform, SameCropTransform, StableColorJitter, StableRotation

class ClothesDataset(data.Dataset):
    '''
    dataset_mode must be 'test' or 'train'
    '''
    def __init__(self, cfg, dataset_mode, device='cpu'):
        super(ClothesDataset).__init__()
        self.cfg = cfg
        self.device = device
        self.data_path = path.join(cfg.dataset_dir, dataset_mode)
        self.transform = transforms.Compose([
            RGBAtoRGBWhiteBlack(),
            MakeSquareWithPad(),
            ToFloatTensor(),
            transforms.Resize((cfg.load_height, cfg.load_width),antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform5 = transforms.Compose([
            RGBAtoRGBWhiteBlack(),
            MakeSquareWithPad(),
            ToFloatTensor(),
            transforms.Resize((cfg.load_height, cfg.load_width),antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5))
        ])
        dataset_list = f'{dataset_mode}_pairs.txt'

        # load data list
        img_names = []
        with open(path.join(cfg.dataset_dir, dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)

        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        hair_color = (254,0,0)
        t_shirt_color = (254,85,0)
        neck_color = (85,51,0)
        botton_color = (0,85,85)
        face_color = (0,0,254)
        debug = False
        # print('Loading image: {}'.format(self.img_names[index]), index)
        img_name = self.img_names[index]
        #horizontal_flip = np.random.random() < self.cfg.horizontal_flip_prob
        #zoom = np.random.random() < self.cfg.crop_prob
        #jitter = np.random.random() < self.cfg.color_jitter_prob
        #angle = np.random.random() < self.cfg.angle_prob
        #random_zoom = SameCropTransform(scale=(self.cfg.min_crop_factor, self.cfg.max_crop_factor))
        #color_jitter = StableColorJitter(self.cfg.brightness, self.cfg.contrast, self.cfg.saturation, self.cfg.hue)
        #random_rotation = StableRotation(self.cfg.rotation_angle)

        img_torch = io.read_image(os.path.join(self.data_path, 'image', img_name))
        img_torch = img_torch.to(self.device)
        if debug:
            print(f'img_torch: {img_torch.shape}')
        # if horizontal_flip:
        #     img_torch = F.hflip(img_torch)
        # if zoom:
        #     img_torch = random_zoom(img_torch)
        # if jitter:
        #     img_torch = color_jitter(img_torch)
        # if angle:
        #     img_torch = random_rotation(img_torch)

        # img_final = self.transform(img_torch)

        # agnostic_mask = io.read_image(os.path.join(self.data_path, 'agnostic-mask', img_name.replace(".jpg","_mask.png")))
        # agnostic_mask = agnostic_mask[:3, :, :]
        # if horizontal_flip:
        #     agnostic_mask = F.hflip(agnostic_mask)
        # if angle:
        #     agnostic_mask = random_rotation(agnostic_mask)
        # agnostic_mask = self.transform(agnostic_mask)
        mask_body_parts_img = Image.open(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png"))).convert('L')
        mask_body_parts_tensor = torch.squeeze(transforms.ToTensor()(mask_body_parts_img))
        mask_body_parts = self.convert_to_rgb_tensor(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png")))
        mask_body_parts = mask_body_parts.to(self.device)
        reference_parts = self.get_mask_by_color(mask_body_parts,[face_color, neck_color]).to(self.device)
        hide_parts = self.get_mask_by_color(mask_body_parts,[hair_color, botton_color]).to(self.device)
        if debug:
            print(f'mask_body_parts: {mask_body_parts.shape}')

        # if horizontal_flip:
        #     mask_body_parts = F.hflip(mask_body_parts)
        # if zoom:
        #     mask_body_parts = random_zoom(mask_body_parts)
        # if angle:
        #     mask_body_parts = random_rotation(mask_body_parts)
        
        # print(f'Mask body parts shape: {mask_body_parts.shape}')
        # print(f'Img shape: {img.shape}')
        target_colors = [(254, 85, 0)]#, (0, 0, 85), (0,119,220), (85,51,0)]
        mask_tensor = self.get_body_color_mask(mask_body_parts, target_colors, img_torch).to(self.device)
        if debug:
            print(f'mask_tensor: {mask_tensor.shape}')
    
        composed_mask_tensor = torch.cat((mask_tensor.clone(), reference_parts[None,:,:], hide_parts[None,:,:]),dim=0)
        
        mask_tensor = self.transform(mask_tensor)
        composed_mask_tensor  = self.transform5(composed_mask_tensor)
        if debug:
            print(f'mask_tensor: {mask_tensor.shape}')
            print(f'composed_mask_tensor: {composed_mask_tensor.shape}')
        
        
        # mask_body_parts = mask_body_parts[:3, :, :]
        # mask_body_parts = self.transform(mask_body_parts)

        # img_masked_rgb = self.adjust_at_offset(img_pil, offset, mask=mask)
        # img_masked_rgb = self.transform(img_masked_rgb)

        cloth = io.read_image(path.join(self.data_path, 'cloth', img_name))
        cloth = cloth.to(self.device)
        if debug:
            print(f'cloth: {cloth.shape}')

        # if horizontal_flip:
        #     cloth = F.hflip(cloth)
        # if jitter:
        #     cloth = color_jitter(cloth)

        cloth_mask = io.read_image(path.join(self.data_path, 'cloth-mask', img_name))
        cloth_mask = cloth_mask.to(self.device)
        if debug:
            print(f'cloth_mask: {cloth_mask.shape}')

        target = ApplyMaskTransform()(cloth, cloth_mask).to(self.device)
        if debug:
            print(f'target: {target.shape}')

        # cloth = self.transform(cloth)
        cloth_mask = SingleChannelToRGBTransform()(cloth_mask)
        cloth_mask = self.transform(cloth_mask)
        if debug:
            print(f'cloth_mask: {cloth_mask.shape}')
        target = self.transform(target)
        if debug:
            print(f'target: {target.shape}')


        result = {
            'img_name': img_name,
            # 'img': img_final,
            # 'img_masked': img_masked_rgb,
            # 'agnostic_mask': agnostic_mask,
            # 'mask_body': mask_tensor,
            'mask_body_parts': mask_body_parts,
            'centered_mask_body': mask_tensor,
            'composed_centered_mask_body': composed_mask_tensor,
            'full_centered_mask_body': mask_body_parts_tensor,
            # 'cloth': cloth,
            #'cloth_mask': cloth_mask,
            'reference_mask': reference_parts,
            'hide_mask': hide_parts,
            'target': target
        }
        return result
    
    @staticmethod
    def get_mask_by_color(img_tensor, colors):
        black = torch.tensor([0,0,0]).to(torch.uint8)
        white = torch.tensor([255, 255, 255]).to(torch.uint8)        
        #print(img_tensor.max())
        device = img_tensor.device
        result = torch.zeros(img_tensor.shape[1], img_tensor.shape[2], 3, dtype=torch.uint8, device=device)
        #data = (img_tensor.clone() * 255).to(torch.uint8).permute(1,2,0)
        #print(data.shape)
        #data[torch.any(data[:,:] != black)] = black
        for color in colors:
            color = torch.tensor(color).to(torch.uint8)
            data = (img_tensor.clone() * 255).to(torch.uint8).permute(1,2,0)
            condition = torch.any(data != color, dim=-1)
            data[condition] = black
            condition = torch.all(data == color, dim=-1)
            data[condition] = white
            result += data
        return result.permute(2,0,1)[0]
    
    @staticmethod
    def get_body_color_mask(mask_body, target_colors, img):
        device = mask_body.device
        target_colors = torch.tensor(target_colors, device=device, dtype=torch.uint8)
        data = (mask_body * 255).to(torch.uint8)
        data = data.permute(1, 2, 0)
        
        X = data[:, :, None, :3]
        intermediate = torch.all(X == target_colors, dim=-1)
        mask = torch.any(intermediate, dim=-1)

        new_data = torch.zeros(4, img.size(1), img.size(2), dtype=torch.uint8, device=device)
        new_data[:3, :, :] = img[:3, :, :]
        new_data[3, :, :][mask] = 255
        #print(f'new_data1: {new_data.shape}')
        
        new_data = CropAlphaChannelTransform()(new_data)
        #print(f'new_data2: {new_data.shape}')
        new_data = AlphaToBlackTransform()(new_data)
        #print(f'new_data3: {new_data.shape}')
        new_data = PadToShapeTransform(img.shape)(new_data)
        #print(f'new_data4: {new_data.shape}')
        
        return new_data
    
    @staticmethod
    def adjust_at_offset(img, offset, mask=None):
        new_img = Image.new('RGB', img.size)
        new_img.paste(img, offset, mask=mask)
        return new_img

    @staticmethod
    def convert_to_rgb_tensor(image_path):
        image = Image.open(image_path)
        if image.mode == 'P':
            image = image.convert('RGB')
        transform = transforms.ToTensor()
        return transform(image)
        

class ClothesDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=1):
        super(ClothesDataLoader, self).__init__()

        self.data_loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True, drop_last=True
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
