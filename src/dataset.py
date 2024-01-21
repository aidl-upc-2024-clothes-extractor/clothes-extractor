from torch.utils import data
from os import path
from PIL import Image
from torchvision import transforms


class ClothesDataset(data.Dataset):

    def __init__(self, opt):
        super(ClothesDataset).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.data_path = path.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.load_width),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        with open(path.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)

        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        img_pil = Image.open(path.join(self.data_path, 'image', img_name)).convert('RGB')
        img = transforms.Resize(self.load_width)(img_pil)
        img = self.transform(img)

        agnostic_mask = Image.open(path.join(self.data_path, 'agnostic-mask', img_name.replace(".jpg","_mask.png"))).convert('RGB')
        agnostic_mask = transforms.Resize(self.load_width)(agnostic_mask)
        agnostic_mask = self.transform(agnostic_mask)

        mask_body = Image.open(path.join(self.data_path, 'image-parse-v3', img_name.replace(".jpg",".png"))).convert('RGBA')
        mask, mask_body = self.get_body_color_mask(mask_body)
        mask_body = transforms.Resize(self.load_width)(mask_body)
        mask_body = self.transform(mask_body)

        img_masked = Image.composite(img_pil, Image.new('RGBA', img_pil.size), mask)
        img_masked_rgb = img_masked.convert('RGB')
        img_masked_rgb = transforms.Resize(self.load_width)(img_masked_rgb)
        img_masked_rgb = self.transform(img_masked_rgb)

        cloth = Image.open(path.join(self.data_path, 'cloth', img_name)).convert('RGB')
        cloth = transforms.Resize(self.load_width)(cloth)
        cloth = self.transform(cloth)

        cloth_mask = Image.open(path.join(self.data_path, 'cloth-mask', img_name)).convert('RGB')
        cloth_mask = transforms.Resize(self.load_width)(cloth_mask)
        cloth_mask = self.transform(cloth_mask)

        result = {
            'img_name': img_name,
            'img': img,
            'img_masked': img_masked_rgb,
            'agnostic_mask': agnostic_mask,
            'mask_body': mask_body,
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

        result = Image.composite(mask_body, Image.new('RGBA', mask_body.size, transparent), mask)

        result_rgb = result.convert('RGB')

        return mask, result_rgb


class ClothesDataLoader:
    def __init__(self, opt, dataset):
        super(ClothesDataLoader, self).__init__()

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=True,
                num_workers=opt.workers, pin_memory=True, drop_last=True
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
