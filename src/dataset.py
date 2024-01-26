from torch.utils import data
from os import path
from PIL import Image
import numpy as np
from torchvision import transforms


class ClothesDataset(data.Dataset):

    def __init__(self, dataset_dir, dataset_mode, load_height, load_width, vertical_flip_prob=0.5):
        super(ClothesDataset).__init__()
        self.load_height = load_height
        self.load_width = load_width
        self.horizontal_flip_prob = vertical_flip_prob
        self.data_path = path.join(dataset_dir, dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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
        # print('Loading image: {}'.format(self.img_names[index]))
        img_name = self.img_names[index]
        horizontal_flip = np.random.random() < self.horizontal_flip_prob

        img_pil = Image.open(path.join(self.data_path, 'image', img_name)).convert('RGB')
        if horizontal_flip:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img_pil)

        agnostic_mask = Image.open(path.join(self.data_path, 'agnostic-mask', img_name.replace(".jpg","_mask.png"))).convert('RGB')
        if horizontal_flip:
            agnostic_mask = agnostic_mask.transpose(Image.FLIP_LEFT_RIGHT)
        agnostic_mask = self.transform(agnostic_mask)

        mask_body_parts = Image.open(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png"))).convert('RGBA')
        if horizontal_flip:
            mask_body_parts = mask_body_parts.transpose(Image.FLIP_LEFT_RIGHT)
        mask, mask_body = self.get_body_color_mask(mask_body_parts)
        centered_mask_body, offset = self.center_masked_area(mask_body, mask)
        mask_body = self.transform(mask_body)
        centered_mask_body = self.transform(centered_mask_body)
        mask_body_parts = self.transform(mask_body_parts.convert('RGB'))

        img_masked_rgb = self.adjust_at_offset(img_pil, offset, mask=mask)
        img_masked_rgb = self.transform(img_masked_rgb)

        cloth = Image.open(path.join(self.data_path, 'cloth', img_name)).convert('RGB')
        if horizontal_flip:
            cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
        cloth = self.transform(cloth)

        cloth_mask = Image.open(path.join(self.data_path, 'cloth-mask', img_name)).convert('RGB')
        cloth_mask = self.transform(cloth_mask)

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
        }
        return result
    
    @staticmethod
    def get_body_color_mask(mask_body):
        target_color = (254, 85, 0)  # #fe5500 in RGB

        opaque = (255, 255, 255, 255)  # White, fully opaque
        transparent = (255, 255, 255, 0)  # White, fully transparent

        data = mask_body.getdata()
        new_data = []
        for item in data:
            new_data.append(opaque if item[:3] == target_color else transparent)

        mask = Image.new('RGBA', mask_body.size)
        mask.putdata(new_data)

        result = Image.new('RGB', mask_body.size)
        result.paste(mask_body, mask=mask)
        result_rgb = result

        return mask, result_rgb
    
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
