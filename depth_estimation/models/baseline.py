
import torch.nn as nn
import torch

def conv_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
    )


def conv_conv_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2), nn.ReLU()
    )

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_1 = conv_conv_2(3, 16)
        self.down_2 = conv_conv_2(16, 32)
        self.down_3 = conv_conv_2(32, 64)
        self.down_4 = conv_conv_2(64, 128)

        self.bottom = conv_conv(128, 128)

        self.up_1 = conv_conv(128, 64)
        self.up_2 = conv_conv(64, 32)
        self.up_3 = conv_conv(32, 16)

        self.conv_final = nn.Conv2d(16, 3, 1, padding=0)

        self.upsample_0 = torch.nn.Upsample(scale_factor=2)
        self.upsample_1 = torch.nn.Upsample(scale_factor=2)
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)
        self.upsample_3 = torch.nn.Upsample(scale_factor=2)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)

        x = self.upsample_0(self.bottom(x))
        x = self.upsample_1(self.up_1(x))
        x = self.upsample_2(self.up_2(x))
        x = self.upsample_3(self.up_3(x))

        return self.conv_final(x)