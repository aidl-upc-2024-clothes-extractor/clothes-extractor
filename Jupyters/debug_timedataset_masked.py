# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: aidl_project
#     language: python
#     name: python3
# ---

# +
import os
from pathlib import Path

# current_folder = globals()['_dh'][0]
current_folder = os.getcwd()
parent = Path(current_folder).parent
os.chdir(parent)
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
# -

import src.dataset as dataset
import importlib
importlib.reload(dataset)

from src.dataset import ClothesDataLoader, ClothesDataset
from src.config import Config
from dataclasses import dataclass
import matplotlib.pyplot as plt

# !pwd

# +
cfg = Config()
cfg.dataset_dir = "./datasets/zalando-hd-resized"
cfg.dataset_mode = "train"
cfg.batch_size = 1
cfg.load_height = 1024
cfg.load_width = 768
cfg.color_jitter_prob = 0


clothes_dataset = ClothesDataset(
    cfg=cfg,
    dataset_mode=cfg.dataset_mode
)
clothes_loader = ClothesDataLoader(
    dataset=clothes_dataset,
    batch_size=cfg.batch_size,
    shuffle=False
)

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()

# +
for i in range(100):
    result = clothes_dataset[i]
    print('Imaname: ', result["img_name"])
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# print('Img shape: ', result["img"].shape)
# print('Cloth shape: ', result["cloth"].shape)
# print('Cloth Mask: ', result["cloth_mask"].shape)
# print('Agnostic Mask: ', result["agnostic_mask"].shape)

# Similarly, we can sample a BATCH from the dataloader by running over its iterator
# iter_ = iter(clothes_loader)
# bimg, blabel = next(iter_)
# print('Batch Img shape: ', bimg.shape)
# print('Batch Label shape: ', blabel.shape)
# print('Batch Img shape: ', bimg.shape)
# print('Batch Label shape: ', blabel.shape)
# print(f'The Batched tensors return a collection of {bimg.shape[0]} grayscale images \
# ({bimg.shape[1]} channel, {bimg.shape[2]} height pixels, {bimg.shape[3]} width \
# pixels)')
# print(f'In the case of the labels, we obtain {blabel.shape[0]} batched integers, one per image')

# +
# image_keys = ["img", "cloth", "cloth_mask", "predict", "agnostic_mask", "mask_body_parts", "mask_body", "centered_mask_body", "img_masked"]
# fig, axes = plt.subplots(1, len(image_keys), figsize=(20, 20))

# for ax, key in zip(axes, image_keys):
#     ax.imshow(result[key].permute(1, 2, 0))
#     ax.axis('off')
#     ax.set_title(key, rotation=90, fontsize=10)
# # -




