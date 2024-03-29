import os
from pathlib import Path
from PIL import Image
import PIL
import tqdm
import argparse

folders = {
    "train": {
        "images": [
            "./data/zalando-hd-resized/train/image",
            "./data/zalando-low-res/train/image",
        ],
        "image_parse_v3": [
            "./data/zalando-hd-resized/train/image-parse-v3",
            "./data/zalando-low-res/train/image-parse-v3",
        ],
        "cloth": [
            "./data/zalando-hd-resized/train/cloth",
            "./data/zalando-low-res/train/cloth",
        ],
        "cloth_mask": [
            "./data/zalando-hd-resized/train/cloth-mask",
            "./data/zalando-low-res/train/cloth-mask",
        ],
        "image_parse_agnostic_v3.2": [
            "./data/zalando-hd-resized/train/image-parse-agnostic-v3.2",
            "./data/zalando-low-res/train/image-parse-agnostic-v3.2",
        ],
    },
    "test": {
        "images": [
            "./data/zalando-hd-resized/test/image",
            "./data/zalando-low-res/test/image",
        ],
        "image_parse_v3": [
            "./data/zalando-hd-resized/test/image-parse-v3",
            "./data/zalando-low-res/test/image-parse-v3",
        ],
        "cloth": [
            "./data/zalando-hd-resized/test/cloth",
            "./data/zalando-low-res/test/cloth",
        ],
        "cloth_mask": [
            "./data/zalando-hd-resized/test/cloth-mask",
            "./data/zalando-low-res/test/cloth-mask",
        ],
        "image_parse_agnostic_v3.2": [
            "./data/zalando-hd-resized/test/image-parse-agnostic-v3.2",
            "./data/zalando-low-res/test/image-parse-agnostic-v3.2",
        ],
    },
}


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--files", type=int, default=None)
    parser.add_argument("-s", "--size", type=int, default=64)

    opt = parser.parse_args()
    return opt


def create_folder(folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def add_margin(pil_img, color=(0, 0, 0)):
    # We are going to create a squared image with the appropriate side and color and the we will
    # paste the image over it
    w, h = pil_img.size
    new_side = max(w, h)
    result = Image.new("RGB", (new_side, new_side), color)
    left = 0
    top = 0
    if w > h:
        # In case the image is wider than it is long we will center it vertically
        top = int((w - h) / 2)
    if h > w:
        # In the event that the image is longer than it is wide we will center it horizontally
        left = int((h - w) / 2)
    result.paste(pil_img, (left, top))
    return result


def downsample_image(src_img: Image, img_side: int = 64) -> Image:
    dst_width = img_side
    dst_height = img_side
    w, h = src_img.size
    # the longer side is img_side the other is modified proportionally
    if w > h:
        dst_height = int(h / w * img_side)
    elif h > w:
        dst_width = int(w / h * img_side)
    # We add a padding before returning the image resized
    return add_margin(
        src_img.resize((dst_width, dst_height), PIL.Image.Resampling.LANCZOS), (0, 0, 0)
    )


def downsample_folder(
    src_dir: str,
    dst_dir: str,
    max_files: int = None,
    area: str = "",
    img_side: int = 64,
) -> None:
    p_src = Path(src_dir)
    count = 0
    files = sorted(p_src.glob("*"))
    # Wehn we have a maximum of file to convert we take a subset of all the files in the folder
    if max_files is not None:
        files = files[:max_files]
    for f in tqdm.tqdm(files, desc=area):
        count += 1
        # read the source image
        image = Image.open(f)
        # decrease resolution
        new_image = downsample_image(image, img_side=img_side)
        dst = dst_dir + "/" + f.name
        # Save destination image
        new_image.save(dst)


def main(opt):
    for stage in folders:  # normally Train or Test
        for key, data in folders[stage].items():
            src = data[0]
            dst = data[1]
            if not Path(src).exists():
                continue
            if not Path(dst).exists():
                create_folder(dst)
            downsample_folder(
                src_dir=src,
                dst_dir=dst,
                max_files=opt.files,
                area=stage + " -> " + key,
                img_side=opt.size,
            )


if __name__ == "__main__":
    opt = get_opt()
    main(opt)
