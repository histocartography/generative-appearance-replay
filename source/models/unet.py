import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution block."""

    def __init__(self, ch_in, ch_out):
        """Initialize convolution block.

        Args:
            ch_in: number of input channels
            ch_out: number of output channels
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """Up-convolution block."""

    def __init__(self, ch_in, ch_out):
        """Initialize up-convolution block.

        Args:
            ch_in: number of input channels
            ch_out: number of output channels
        """
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """U-Net."""

    def __init__(self, in_channels=1, classes=1):
        """Initialize U-Net.

        Args:
            in_channels: number of input channels. Defaults to 1.
            classes: number of segmentation classes. Defaults to 1.
        """
        super(U_Net, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=in_channels, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1
