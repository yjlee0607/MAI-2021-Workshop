import torch

from pathlib import Path
from models.baseline import UNet
from utils.dataloading import BasicDataset

dir_img = Path('./challenge_data/train/rgb/')
dir_mask = Path('./challenge_data/train/depth/')

dataset = BasicDataset(images_dir=dir_img,
                       mask_dir=dir_mask,
                       scale=1,
                       )

print('hi')