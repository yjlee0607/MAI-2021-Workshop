import torch
import torchvision
from torch.utils.data import DataLoader

from utils.dataloading import BasicDataset

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((160,128))
])

dataset = BasicDataset(images_dir='challenge_data/train/depth',
                       mask_dir='challenge_data/train/rgb',
                       scale=1,
                       transform=transform)



train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)