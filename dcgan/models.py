import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        # HWC: 1x1x100 -> 64x64x3 (100-dimensional code vector to 64x64 3-channel images).
        self.main = nn.Sequential(
            # HWC: 1x1x100 -> 4x4x1024
            # W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 4x4x1024 -> 8x8x512
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, output_padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8x512 -> 16x16x256
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, output_padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, bias=False, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32x128 -> 64x64x3
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=7, padding=1, bias=False, dilation=6),
            nn.Tanh()
        )

        def forward(self, z):
            return self.main(z)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        # HWC: 64x64x3 -> 1 number in [0,1] (probability for binary classification)
        self.main = nn.Sequential(
            # 64x64x3 -> 32x32x128
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            # 32x32x128 -> 16x16x256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 16x16x256 -> 8x8x512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 8x8x512 -> 4x4x1024
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),

            # 4x4x1024 -> 1
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, Gz):
        return self.main(Gz)
    



