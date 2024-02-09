import os
from pathlib import Path
from PIL import Image
import PIL
import tqdm
import argparse
import numpy as np

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--max_files', type=int, default=None)
    parser.add_argument('--dst_file', type=str, default='./data/out_color.txt')
    parser.add_argument('--src_dir', type=str, default='./data/zalando-hd-resized')

    opt = parser.parse_args()
    return opt


def check_image_contains_color(img: Image, color: tuple) -> bool:
    data = np.array(img)
    data[(data != color).any(axis = -1)] = (0,0,0)
    data[(data == color).all(axis = -1)] = (1,1,1)
    data = data[:,:,0]
    return (np.sum(data) > 0)


def check_color_folder(src_dir: str, max_files: int  = 100) -> None:
    orange_color = (254,85,0)
    p_src = Path(src_dir)
    count = 0
    files = sorted(p_src.glob('*'))
    result = []
    if max_files is not None:
        files = files[:max_files]
    # When we have a maximum of file to convert we take a subset of all the files in the folder
    for f in tqdm.tqdm(files):
        #read the source image
        image = Image.open(f).convert('RGB')
        if check_image_contains_color(image, orange_color):
            #decrease resolution
            result.append(f.name)
        count += 1
    return result

def write_pair_file(p: str = '.', files: list = []) -> None:
    with open(p, "w") as txt_file:
        for fname in files:
            txt_file.write(fname + " " + fname + "\n")

def main(opt):
    files = check_color_folder(opt.src_dir, opt.max_files)
    write_pair_file(opt.dst_file, files)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)
