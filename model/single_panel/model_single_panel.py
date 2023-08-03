import torch
import torch.nn as nn
import numpy as np

from model.custom_conv import CubeSpherePadding2D, CubeSphereConv2D
from model.single_panel.custom_conv_single_panel import CubeSpherePadding2DSinglePanel, CubeSphereConv2DSinglePanel

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py


class Discriminator(nn.Module):
    def __init__(self, nbins: int):
        super(Discriminator, self).__init__()
        self.nbins = nbins
        self.features = nn.Sequential(
            # input size. (nbin) x 5 x 16 x 16
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(self.nbins, 64, (3, 3), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 5 x 16 x 16
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(64, 64, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 5 x 16 x 16
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(64, 128, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (nbins) x 5 x (hrtf_size) x (hrtf_size)
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(128, 128, (3, 3), (2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(128, 256, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 5 x 8 x 8
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(256, 256, (3, 3), (2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(256, 512, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 5 x 4 x 4
            # CubeSpherePadding2D(1),
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
        # print('out-flatten')
        # print(np.shape(out.detach().numpy()))
        out = torch.flatten(out, 1)
        # print('out-flatten')
        # print(np.shape(out.detach().numpy()))
        out = self.classifier(out)
        # print('out-flatten')
        # print(np.shape(out.detach().numpy()))

        return out


class ResidualConvBlockSinglePanel(nn.Module):
    """Implements residual conv function.
    Args:
        channels (int): Number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlockSinglePanel, self).__init__()
        self.rcb = nn.Sequential(
            # CubeSpherePadding2DSinglePanel(1),
            CubeSphereConv2DSinglePanel(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            # CubeSpherePadding2DSinglePanel(1),
            CubeSphereConv2DSinglePanel(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # print('identity:')
        # print(np.shape(identity.detach().numpy()))
        # print('rcb:')
        out = self.rcb(x)
        # print('out:')
        # print(np.shape(out.detach().numpy()))
        out = torch.add(out, identity)

        return out


class UpsampleBlockSinglePanel(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsampleBlockSinglePanel, self).__init__()
        self.upsample_block_1 = nn.Sequential(
            # CubeSpherePadding2D(1),
            CubeSphereConv2DSinglePanel(channels, channels * 4, (3, 3), (1, 1))
        )
        self.upsample_block_2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.upsample_block_1(x)
        # print('out1-upsample:')
        # print(np.shape(out1.detach().numpy()))
        out = self.upsample_block_2(out1)
        # print('out-upsample:')
        # print(np.shape(out.detach().numpy()))

        return out


class GeneratorSinglePanel(nn.Module):
    def __init__(self, upscale_factor: int, nbins: int) -> None:
        super(GeneratorSinglePanel, self).__init__()
        self.nbins = nbins
        self.ngf = 512
        self.num_upsampling_blocks = int(np.log(upscale_factor)/np.log(2))

        # First conv layer.
        self.conv_block1 = nn.Sequential(
            # CubeSpherePadding2DSinglePanel(1),
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
            # CubeSpherePadding2DSinglePanel(1),
            CubeSphereConv2DSinglePanel(self.ngf, self.ngf, (3, 3), (1, 1), bias=False),
            nn.BatchNorm2d(self.ngf),
        )

        # Upscale block
        upsampling = []
        for _ in range(self.num_upsampling_blocks):
            upsampling.append(UpsampleBlockSinglePanel(self.ngf))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Sequential(
            # CubeSpherePadding2DSinglePanel(1),
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
        # print('GeneratorSinglePanel End')

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