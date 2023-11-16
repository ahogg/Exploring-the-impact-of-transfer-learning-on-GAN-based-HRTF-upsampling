import torch
import torch.nn as nn
import numpy as np

from model.custom_conv import CubeSpherePadding2D, CubeSphereConv2D
from model.single_panel.custom_conv_single_panel import CubeSphereConv2DSinglePanel

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py

class Discriminator(nn.Module):
    def __init__(self, nbins: int):
        super(Discriminator, self).__init__()
        self.nbins = nbins
        self.features = nn.Sequential(

            CubeSphereConv2DSinglePanel(self.nbins, 64, (3, 3), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),

            CubeSphereConv2DSinglePanel(64, 64, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            CubeSphereConv2DSinglePanel(64, 128, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            CubeSphereConv2DSinglePanel(128, 128, (3, 3), (2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            CubeSphereConv2DSinglePanel(128, 256, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            CubeSphereConv2DSinglePanel(256, 256, (3, 3), (2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            CubeSphereConv2DSinglePanel(256, 512, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            CubeSphereConv2DSinglePanel(512, 512, (3, 3), (2, 2), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*8*3, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ResidualConvBlockSinglePanel(nn.Module):
    """Implements residual conv function.
    Args:
        channels (int): Number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlockSinglePanel, self).__init__()
        self.rcb = nn.Sequential(
            CubeSphereConv2DSinglePanel(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            CubeSphereConv2DSinglePanel(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)

        return out


class UpsampleBlockSinglePanel(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlockSinglePanel, self).__init__()
        self.upsample_block_1 = nn.Sequential(
            CubeSphereConv2DSinglePanel(channels, channels * 4, (3, 3), (1, 1))
        )
        self.upsample_block_2 = nn.Sequential(
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.upsample_block_1(x)
        out = self.upsample_block_2(out1)

        return out


class GeneratorSinglePanel(nn.Module):
    def __init__(self, upscale_factor: int, nbins: int) -> None:
        super(GeneratorSinglePanel, self).__init__()
        self.nbins = nbins
        self.ngf = 512
        self.num_upsampling_blocks = int(np.log(upscale_factor)/np.log(2))

        # First conv layer.
        self.conv_block1 = nn.Sequential(
            CubeSphereConv2DSinglePanel(self.nbins, self.ngf, (3, 3), (1, 1)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(8):
            trunk.append(ResidualConvBlockSinglePanel(self.ngf))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            CubeSphereConv2DSinglePanel(self.ngf, self.ngf, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(self.ngf),
        )

        # Upscale block
        upsampling = []

        if self.num_upsampling_blocks == 4:
            upsampling.append(torch.nn.Upsample(scale_factor=(2, 1.5)))
            self.num_upsampling_blocks -= 1
        elif self.num_upsampling_blocks == 5:
            upsampling.append(torch.nn.Upsample(scale_factor=(4, 3)))
            self.num_upsampling_blocks -= 2
        elif self.num_upsampling_blocks == 6:
            print('NOT DONE')

        for _ in range(self.num_upsampling_blocks):
            upsampling.append(UpsampleBlockSinglePanel(self.ngf, 2))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Sequential(
            CubeSphereConv2DSinglePanel(self.ngf, self.nbins, (3, 3), (1, 1))
        )

        self.classifier = nn.Softplus()

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # print('GeneratorSinglePanel Start')
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, CubeSphereConv2DSinglePanel):
                nn.init.kaiming_normal_(module.equatorial_weight)
                nn.init.kaiming_normal_(module.polar_weight)
                if module.equatorial_bias is not None:
                    nn.init.constant_(module.equatorial_bias, 0)
                    nn.init.constant_(module.polar_bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)