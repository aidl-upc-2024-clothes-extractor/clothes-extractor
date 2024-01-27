from torch.utils import data
from os import path
import numpy as np
from torchvision import transforms, io
from torchvision.transforms import functional as F
import torch
import os
from PIL import Image

class SameCropTransform:
    def __init__(self, output_size, scale=(0.8, 1.0)):
        self.output_size = output_size
        self.scale = scale
        self.i = None
        self.j = None
        self.h = None
        self.w = None

    def __call__(self, img):
        if self.i is None or self.j is None or self.h is None or self.w is None:
            self.i, self.j, self.h, self.w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=(1.0, 1.0))
            
        img_resized = F.resized_crop(img, self.i, self.j, self.h, self.w, self.output_size)

        return img_resized

class StableColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.fn_idx, self.brightness, self.contrast, self.saturation, self.hue = transforms.ColorJitter.get_params(
            jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
        )

    def __call__(self, img):
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness != 0:
                brightness_factor = self.brightness
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and self.contrast != 0:
                contrast_factor = self.contrast
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and self.saturation != 0:
                saturation_factor = self.saturation
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and self.hue != 0:
                hue_factor = self.hue
                img = F.adjust_hue(img, hue_factor)

        return img

class StableRotation:
    def __init__(self, angle=0):
        self.angle = np.random.uniform(-angle, angle)

    def __call__(self, img):
        return F.rotate(img, self.angle)

class MakeSquareWithPad:
    def __init__(self, fill=0):
        self.fill = fill  # fill color, 0 for black

    def __call__(self, img):
        _, width, height = img.size()
        max_side = max(width, height)
        padding_left = padding_right = padding_top = padding_bottom = 0

        if width < max_side:
            padding_left = (max_side - width) // 2
            padding_right = max_side - width - padding_left
        elif height < max_side:
            padding_top = (max_side - height) // 2
            padding_bottom = max_side - height - padding_top

        padding = (padding_left, padding_top, padding_right, padding_bottom)
        return F.pad(img, padding, fill=self.fill, padding_mode='constant')


class ToFloatTensor(object):
    def __call__(self, tensor):
        if tensor.dtype.is_floating_point:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        return tensor.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'



class ClothesDataset(data.Dataset):

    def __init__(
            self,
            dataset_dir, dataset_mode,
            load_height, load_width,
            horizontal_flip_prob=0.5,
            rotation_prob=0.5, rotation_angle=10,
            crop_prob=0.5, min_crop_factor=0.65, max_crop_factor=0.92,
            brightness=0.15, contrast=0.3, saturation=0.3, hue=0.05,
            color_jitter_prob=1,
            angle_prob=0.5, angle=10
    ):
        super(ClothesDataset).__init__()
        self.load_height = load_height
        self.load_width = load_width
        self.horizontal_flip_prob = horizontal_flip_prob
        self.crop_prob = crop_prob
        self.min_crop_factor = min_crop_factor
        self.max_crop_factor = max_crop_factor
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.angle_prob = angle_prob
        self.angle = angle
        self.rotation_prob = rotation_prob
        self.rotation_angle = rotation_angle
        self.data_path = path.join(dataset_dir, dataset_mode)
        self.transform = transforms.Compose([
            MakeSquareWithPad(),
            ToFloatTensor(),
            transforms.Resize(self.load_width),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_list = f'{dataset_mode}_pairs.txt'

        # load data list
        img_names = []
        with open(path.join(dataset_dir, dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)

        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        print('Loading image: {}'.format(self.img_names[index]), index)
        img_name = self.img_names[index]
        horizontal_flip = np.random.random() < self.horizontal_flip_prob
        zoom = np.random.random() < self.crop_prob
        jitter = np.random.random() < self.color_jitter_prob
        angle = np.random.random() < self.angle_prob
        random_zoom = SameCropTransform((self.load_height, self.load_width), scale=(self.min_crop_factor, self.max_crop_factor))
        color_jitter = StableColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        random_rotation = StableRotation(self.rotation_angle)

        img_pil = io.read_image(os.path.join(self.data_path, 'image', img_name))

        if horizontal_flip:
            img_pil = F.hflip(img_pil)
        if zoom:
            img_pil = random_zoom(img_pil)
        if jitter:
            img_pil = color_jitter(img_pil)
        if angle:
            img_pil = random_rotation(img_pil)

        img = self.transform(img_pil)

        agnostic_mask = io.read_image(os.path.join(self.data_path, 'agnostic-mask', img_name.replace(".jpg","_mask.png")))
        agnostic_mask = agnostic_mask[:3, :, :]
        if horizontal_flip:
            agnostic_mask = F.hflip(agnostic_mask)
        if angle:
            agnostic_mask = random_rotation(agnostic_mask)
        agnostic_mask = self.transform(agnostic_mask)

        mask_body_parts = self.convert_to_rgb_tensor(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png")))
        print(mask_body_parts.shape)
        if horizontal_flip:
            mask_body_parts = F.hflip(mask_body_parts)
        if zoom:
            mask_body_parts = random_zoom(mask_body_parts)
        if angle:
            mask_body_parts = random_rotation(mask_body_parts)

        target_colors = [(254, 85, 0), (0, 0, 85), (0,119,220), (85,51,0)]
        mask, mask_body = self.get_body_color_mask(mask_body_parts, target_colors)
        # centered_mask_body, offset = self.center_masked_area(mask_body, mask)
        # mask_body = self.transform(mask_body)
        # centered_mask_body = self.transform(centered_mask_body)
        mask_body_parts = mask_body_parts[:3, :, :]
        # mask_body_parts = self.transform(mask_body_parts)

        # img_masked_rgb = self.adjust_at_offset(img_pil, offset, mask=mask)
        # img_masked_rgb = self.transform(img_masked_rgb)

        # cloth = io.read_image(path.join(self.data_path, 'cloth', img_name))
        # if horizontal_flip:
        #     cloth = F.hflip(cloth)
        # if jitter:
        #     cloth = color_jitter(cloth)

        # cloth_mask = io.read_image(path.join(self.data_path, 'cloth-mask', img_name))
        # cloth_color_mask, _ = self.get_body_color_mask(cloth_mask, [(255, 255, 255)])
        # predict = self.adjust_at_offset(cloth, None, mask=cloth_color_mask)

        # cloth = self.transform(cloth)
        # cloth_mask = cloth_mask[:3, :, :]
        # cloth_mask = self.transform(cloth_mask)
        # predict = self.transform(predict)


        result = {
            'img_name': img_name,
            'img': img,
            # 'img_masked': img_masked_rgb,
            'agnostic_mask': agnostic_mask,
            'mask_body': mask_body,
            'mask_body_parts': mask_body_parts,
            # 'centered_mask_body': centered_mask_body,
            # 'cloth': cloth,
            # 'cloth_mask': cloth_mask,
            # 'predict': predict
        }
        return result
    
        
    @staticmethod
    def get_body_color_mask(mask_body, target_colors):
        device = "cpu"
        t = transforms.ToPILImage()
        mask_body = t(mask_body.cpu())
        target_colors = torch.tensor(target_colors, device=device, dtype=torch.uint8)
        data = torch.tensor(np.array(mask_body), device=device, dtype=torch.uint8)

        mask = torch.any(torch.all(data[:, :, None, :3] == target_colors, dim=-1), dim=-1)

        new_data = torch.zeros(mask_body.size[1], mask_body.size[0], 4, dtype=torch.uint8, device=device)
        new_data[:, :, :3] = data[:, :, :3]
        new_data[:, :, 3][mask] = 255

        mask_image = Image.fromarray(new_data.cpu().numpy(), 'RGBA')
        result = Image.new('RGB', mask_body.size)
        result.paste(mask_body, mask=mask_image)

        t = transforms.ToTensor()
        mask_image = t(mask_image)
        result = t(result)

        return mask_image, result
    
    @staticmethod
    def adjust_at_offset(img, offset, mask=None):
        _, height, width = img.shape
        print(img.shape)
        new_img = torch.zeros_like(img)

        # Applying offset
        dx, dy = offset
        new_img[:, dy:height, dx:width] = img[:, :height-dy, :width-dx]

        # Applying mask if provided
        if mask is not None:
            alpha_mask = mask[-1,:,:].unsqueeze(0).expand_as(new_img)
            alpha_mask = alpha_mask.to(dtype=new_img.dtype)
            new_img *= alpha_mask

        return new_img


    @staticmethod
    def center_masked_area(img, mask):
        mask_alpha = mask[0, :, :].cpu().numpy()

        rows = np.any(mask_alpha, axis=1)
        cols = np.any(mask_alpha, axis=0)
        print(rows.shape, cols.shape)
        print(rows, cols)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        masked_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)

        img_center = (img.size(2) // 2, img.size(1) // 2)

        offset = (img_center[0] - masked_center[0], img_center[1] - masked_center[1])

        new_img = ClothesDataset.adjust_at_offset(img, offset, mask=mask)

        return new_img, offset

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
