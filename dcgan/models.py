import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        # CHW: 100x1x1 -> 3x64x64 (100-dimensional code vector to 64x64 3-channel images).
        self.main = nn.Sequential(
            # CHW: 10x1x1 -> 1024x4x4
            # W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 1024x4x4 -> 512x8x8
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, bias=False, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512x8x8 -> 256x16x16
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, bias=False, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256x16x16 -> 128x32x32
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, bias=False, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128x32x32 -> 3x64x64
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=7, padding=1, bias=False, dilation=6),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        # CHW: 3X64X64 -> 1 number in [0,1] (probability for binary classification)
        self.main = nn.Sequential(
            # 3x64x64 -> 128x32x32
            # H_out = floor[(W_in + 2 * padding - dilation * (kernel - 1)) / stride + 1]
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            # 128x32x32 -> 256x16x16
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 256x16x16 -> 512x8x8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 512x8x8 -> 1024x4x4
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),

            # 1024x4x4 -> 1
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, Gz):
        return self.main(Gz)
    



