from pathlib import Path
import numpy as np
import random

import torch
import torchvision
from torch.utils.data import DataLoader

from utils.dataloading import MaiDataset
from models.baseline import UNet

# random seed control
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def silog_loss(gt,pred):
    gt[gt<1]=1
    pred[pred<1]=1
    gt.to(torch.float)
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    return silog
############################
dir_img = Path('./challenge_data/train/rgb/')
dir_mask = Path('./challenge_data/train/depth/')


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((160,128)),
    torchvision.transforms.ToTensor()
])

dataset = MaiDataset(root_dir='./challenge_data/train',
                    mode='train',
                    transform=transform)


train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

device='cuda'
model = UNet()
model.to(device)

epochs = 5
criteria = silog_loss
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, eps=1e-3,)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epochs)

torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
    
    for idx,sample in enumerate(train_dataloader):
        rgb_image = sample['rgb_image'].to(device)
        depth_image = sample['depth_image'].to(torch.float).to(device)#.to(torch.int32).to(device)
        
        output = model(rgb_image) * 1000.
        output = torch.clip(output,min=1.0,max=65535.0)
        output = torch.clip(output,min=0.0,max=65535.0)
        # output = output.to(torch.int32)# if device == 'cpu' else output.to(torch.cuda.IntTensor)
        
        # errors = compute_errors(depth_image, output)
        # loss 
        # https://github.com/cleinc/bts/blob/dd62221bc50ff3cbe4559a832c94776830247e2e/pytorch/bts_main.py#L417
        loss = criteria(depth_image, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{idx}/{len(train_dataloader)}] Loss: {loss.data}")
        # with torch.no_grad():
        #     print(compute_errors(depth_image, output))
    scheduler.step()