from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import torch


class SameCropTransform:
    def __init__(self, scale=(0.8, 1.0)):
        self.scale = scale
        self.i = None
        self.j = None
        self.h = None
        self.w = None

    def __call__(self, img):
        if self.i is None or self.j is None or self.h is None or self.w is None:
            self.i, self.j, self.h, self.w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=(1.0, 1.0)
            )

        _, h, w = img.size()
        img_resized = F.resized_crop(img, self.i, self.j, self.h, self.w, (h, w))

        return img_resized


class StableColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.fn_idx, self.brightness, self.contrast, self.saturation, self.hue = (
            transforms.ColorJitter.get_params(
                jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
            )
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
        _, height, width = img.size()
        max_side = max(width, height)
        padding_left = padding_right = padding_top = padding_bottom = 0

        if width < max_side:
            padding_left = (max_side - width) // 2
            padding_right = max_side - width - padding_left
        elif height < max_side:
            padding_top = (max_side - height) // 2
            padding_bottom = max_side - height - padding_top

        padding = (padding_left, padding_top, padding_right, padding_bottom)
        return F.pad(img, padding, fill=self.fill, padding_mode="constant")


class ToFloatTensor(object):
    def __call__(self, tensor):
        if tensor.dtype.is_floating_point:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        return tensor.float()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RGBAtoRGBWhiteBlack:

    def __call__(self, img):
        if img.shape[0] == 4:
            rgb_img, alpha_img = img[:3, :, :], img[3:4, :, :]

            white = torch.ones_like(rgb_img)
            black = torch.zeros_like(rgb_img)

            rgb_transformed = torch.where(alpha_img > 0, white, black)

            return rgb_transformed
        else:
            return img


class SingleChannelToRGBTransform:
    def __call__(self, img):
        if img.shape[0] != 1:
            raise ValueError("Image must have a single channel")

        return img.repeat(3, 1, 1)


class CropAlphaChannelTransform:
    def __call__(self, img):
        alpha_channel = img[-1]

        non_zero_rows = torch.any(alpha_channel != 0, dim=1)
        non_zero_cols = torch.any(alpha_channel != 0, dim=0)

        # Crop the image
        cropped_img = img[:, non_zero_rows, :]
        cropped_img = cropped_img[:, :, non_zero_cols]

        return cropped_img


class AlphaToBlackTransform:
    def __call__(self, img):
        if img.shape[0] != 4:
            raise ValueError("Image must have 4 channels")

        alpha = img[3].unsqueeze(0)
        rgb = img[:3]

        #rgb_float = rgb.to(torch.float32) / 255.0
        #background = torch.randn_like(rgb_float) * torch.std(rgb_float) + torch.mean(rgb_float)
        #background = (background * 255).to(torch.uint8)
        background = torch.zeros_like(rgb)

        return torch.where(alpha > 0, rgb, background)


class PadToShapeTransform:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, img):
        _, h, w = img.shape
        target_h, target_w = self.output_shape[1], self.output_shape[2]

        pad_vert = max((target_h - h) // 2, 0)
        pad_horiz = max((target_w - w) // 2, 0)

        left = pad_horiz
        top = pad_vert
        right = pad_horiz
        bottom = pad_vert

        if h + 2 * pad_vert < target_h:
            bottom += 1
        if w + 2 * pad_horiz < target_w:
            right += 1

        padded_img = F.pad(img, (left, top, right, bottom))

        if padded_img.shape[1] > target_h or padded_img.shape[2] > target_w:
            padded_img = padded_img[:, :target_h, :target_w]

        return padded_img


class ApplyMaskTransform:
    def __call__(self, img, mask):
        if img.shape[1:] != mask.shape[1:]:
            raise ValueError("RGB image and mask must have the same height and width")

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        black_background = torch.zeros_like(img)
        return torch.where(mask > 0, img, black_background)


def plot_alpha_channel(img):
    from matplotlib import pyplot as plt

    if img.shape[0] < 4:
        raise ValueError("Image does not have an alpha channel")

    alpha_channel = img[-1]

    plt.imshow(alpha_channel, cmap="gray")
    plt.title("Alpha Channel")
    plt.colorbar()
    plt.show()


def plot_alpha_channel_raw(img):

    if len(img.shape) != 3:
        raise ValueError("Image is not 3D")

    if img.shape[0] != 1:
        raise ValueError("Image does not have an alpha channel")

    img = img[0]
    from matplotlib import pyplot as plt

    plt.imshow(img, cmap="gray")
    plt.title("Alpha Channel")
    plt.colorbar()
    plt.show()


def plot_rgb(img):
    from matplotlib import pyplot as plt

    if img.shape[0] < 3:
        raise ValueError("Image does not have an RGB channel")

    rgb = img[:3]

    plt.imshow(rgb.permute(1, 2, 0))
    plt.title("RGB")
    plt.show()
