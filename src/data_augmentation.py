from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np


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
        width, height = img.size
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
