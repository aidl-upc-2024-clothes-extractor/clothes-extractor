from torch.utils import data
from os import path
from PIL import Image
import torch

from src.data_augmentation import *


class ClothesDataset(data.Dataset):
    '''
    dataset_mode must be 'test' or 'train'
    '''
    def __init__(self, cfg, dataset_mode):
        super(ClothesDataset).__init__()
        self.cfg = cfg
        self.data_path = path.join(cfg.dataset_dir, dataset_mode)
        self.transform = transforms.Compose([
            MakeSquareWithPad(),
            transforms.ToTensor(),
            transforms.Resize(self.cfg.load_width),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        print('Loading image: {}'.format(self.img_names[index]), index)
        img_name = self.img_names[index]
        horizontal_flip = np.random.random() < self.cfg.horizontal_flip_prob
        zoom = np.random.random() < self.cfg.crop_prob
        jitter = np.random.random() < self.cfg.color_jitter_prob
        angle = np.random.random() < self.cfg.angle_prob
        random_zoom = SameCropTransform((self.cfg.load_height, self.cfg.load_width), scale=(self.cfg.min_crop_factor, self.cfg.max_crop_factor))
        color_jitter = StableColorJitter(self.cfg.brightness, self.cfg.contrast, self.cfg.saturation, self.cfg.hue)
        random_rotation = StableRotation(self.cfg.rotation_angle)

        img_pil = Image.open(path.join(self.data_path, 'image', img_name)).convert('RGB')
        if horizontal_flip:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        if zoom:
            img_pil = random_zoom(img_pil)
        if jitter:
            img_pil = color_jitter(img_pil)
        if angle:
            img_pil = random_rotation(img_pil)

        img = self.transform(img_pil)

        agnostic_mask = Image.open(path.join(self.data_path, 'agnostic-mask', img_name.replace(".jpg","_mask.png"))).convert('RGB')
        if horizontal_flip:
            agnostic_mask = agnostic_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if angle:
            agnostic_mask = random_rotation(agnostic_mask)
        agnostic_mask = self.transform(agnostic_mask)

        mask_body_parts = Image.open(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png"))).convert('RGBA')
        if horizontal_flip:
            mask_body_parts = mask_body_parts.transpose(Image.FLIP_LEFT_RIGHT)
        if zoom:
            mask_body_parts = random_zoom(mask_body_parts)
        if angle:
            mask_body_parts = random_rotation(mask_body_parts)

        target_colors = [(254, 85, 0), (0, 0, 85), (0,119,220), (85,51,0)]
        mask, mask_body = self.get_body_color_mask(mask_body_parts, target_colors)
        centered_mask_body, offset = self.center_masked_area(mask_body, mask)
        mask_body = self.transform(mask_body)
        centered_mask_body = self.transform(centered_mask_body)
        mask_body_parts = self.transform(mask_body_parts.convert('RGB'))

        img_masked_rgb = self.adjust_at_offset(img_pil, offset, mask=mask)
        img_masked_rgb = self.transform(img_masked_rgb)

        cloth = Image.open(path.join(self.data_path, 'cloth', img_name)).convert('RGB')
        if horizontal_flip:
            cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
        if jitter:
            cloth = color_jitter(cloth)

        cloth_mask = Image.open(path.join(self.data_path, 'cloth-mask', img_name)).convert('RGBA')
        cloth_color_mask, _ = self.get_body_color_mask(cloth_mask, [(255, 255, 255)])
        predict = self.adjust_at_offset(cloth, None, mask=cloth_color_mask)

        cloth = self.transform(cloth)
        cloth_mask = self.transform(cloth_mask.convert('RGB'))
        predict = self.transform(predict)


        result = {
            'img_name': img_name,
            'img': img,
            'img_masked': img_masked_rgb,
            'agnostic_mask': agnostic_mask,
            'mask_body': mask_body,
            'mask_body_parts': mask_body_parts,
            'centered_mask_body': centered_mask_body,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
            'predict': predict
        }
        return result
    
        
    @staticmethod
    def get_body_color_mask(mask_body, target_colors):
        device = "cpu"

        target_colors = torch.tensor(target_colors, device=device, dtype=torch.uint8)
        data = torch.tensor(np.array(mask_body), device=device, dtype=torch.uint8)

        mask = torch.any(torch.all(data[:, :, None, :3] == target_colors, dim=-1), dim=-1)

        new_data = torch.zeros(data.shape, dtype=torch.uint8, device=device)
        new_data[:, :, :3] = data[:, :, :3]
        new_data[:, :, 3][mask] = 255

        mask_image = Image.fromarray(new_data.cpu().numpy(), 'RGBA')
        result = Image.new('RGB', mask_body.size)
        result.paste(mask_body, mask=mask_image)

        return mask_image, result

    
    @staticmethod
    def adjust_at_offset(img, offset, mask=None):
        new_img = Image.new('RGB', img.size)
        new_img.paste(img, offset, mask=mask)
        return new_img

    @staticmethod
    def center_masked_area(img, mask):
        mask_alpha = np.array(mask)[:, :, 3]

        rows = np.any(mask_alpha, axis=1)
        cols = np.any(mask_alpha, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        masked_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)

        img_center = (img.width // 2, img.height // 2)

        offset = (img_center[0] - masked_center[0], img_center[1] - masked_center[1])

        new_img = ClothesDataset.adjust_at_offset(img, offset, mask=mask)

        return new_img, offset
        

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
