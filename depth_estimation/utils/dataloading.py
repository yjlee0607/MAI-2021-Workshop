# https://github.com/OniroAI/MonoDepth-PyTorch/blob/master/data_loader.py

import os
from PIL import Image

from torch.utils.data import Dataset


class MaiDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        # default settings
        self.transform = transform
        self.mode = mode
        
        # data settings
        if mode == 'train':
            rgb_dir = os.path.join(root_dir, 'rgb')
            depth_dir = os.path.join(root_dir, 'depth')
            
            self.rgb_paths = sorted([os.path.join(rgb_dir, fname) for fname in os.listdir(rgb_dir)])
            self.depth_paths = sorted([os.path.join(depth_dir, fname) for fname in os.listdir(depth_dir)])
            assert len(self.rgb_paths) == len(self.depth_paths)
            
        else: # test
            rgb_dir = os.path.join(root_dir, 'rgb')
            self.rgb_paths = sorted([os.path.join(rgb_dir, fname) for fname in os.listdir(rgb_dir)])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx])
        if self.mode == 'train':
            depth_image = Image.open(self.depth_paths[idx])
            

            if self.transform:
                rgb_image = self.transform(rgb_image)
                depth_image = self.transform(depth_image)            
                sample = {'rgb_image': rgb_image, 'depth_image': depth_image}
                return sample
            else:
                return sample
        else:
            if self.transform:
                rgb_image = self.transform(rgb_image)
            return rgb_image


if __name__ == '__main__':
    import torch
    import torchvision

    from torch.utils.data import DataLoader

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((160,128)),
        torchvision.transforms.ToTensor()
    ])

    dataset = MaiDataset(root_dir='../challenge_data/train',
                        mode='train',
                        transform=transform)
    

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    for a in train_dataloader:
        print(a)

    print("hi")
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)