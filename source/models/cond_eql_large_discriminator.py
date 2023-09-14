import torch
from torch import nn as nn
import numpy as np

from .layers import EQConv2d, EQLinear


class Prologue(nn.Module):
    """Class for the prologue of the discriminator."""

    def __init__(self, stage, out_nchan=512):
        """Initialize discriminator prologue.

        Args:
            stage: output stage of the feature extractor serving as input to the discriminator
            out_nchan: number of output channels. Defaults to 512.
        """
        super().__init__()
        # stage of the feature extractor
        self.stage = stage
        # downsampling
        self.biliDown = nn.AvgPool2d(2, stride=2)
        # ceiled downsampling
        self.cbiliDown = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        # activation function
        self.act = nn.LeakyReLU(0.2)
        if self.stage == "multiple":
            base = 64
            self.conv1 = EQConv2d(base, base, kernel_size=3, stride=1, padding=1, bias=True)
            self.ingestion_1 = EQConv2d(3 * base, base, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = EQConv2d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=True)
            self.ingestion_2 = EQConv2d(6 * base, 2 * base, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv3 = EQConv2d(2 * base, 2 * base, kernel_size=3, stride=1, padding=1, bias=True)
            self.ingestion_3 = EQConv2d(10 * base, 2 * base, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv4 = EQConv2d(2 * base, 4 * base, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv5 = EQConv2d(4 * base, 4 * base, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv6 = EQConv2d(4 * base, 8 * base, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv7 = EQConv2d(8 * base, 8 * base, kernel_size=3, stride=1, padding=1, bias=True)
        elif self.stage == "stage0":
            in_n_chan = 1
            base = 32
            # in_n_chanx256x256 -> 32x256x256
            self.conv1 = EQConv2d(in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True)
            # 32x128x128 -> 64x128x128
            self.conv2 = EQConv2d(base, base * 2, kernel_size=3, stride=1, padding=1, bias=True)
            # 64x64x64-> 128x64x64
            self.conv3 = EQConv2d(base * 2, base * 4, kernel_size=3, stride=1, padding=1, bias=True)
            # 128x32x32 -> 256x32x32
            self.conv4 = EQConv2d(base * 4, base * 8, kernel_size=3, stride=1, padding=1, bias=True)
            # 256x16x16 -> 512x16x16
            self.conv5 = EQConv2d(base * 8, base * 16, kernel_size=3, stride=1, padding=1, bias=True)
            # 512x8x8 -> 512x8x8
            self.conv6 = EQConv2d(base * 16, base * 16, kernel_size=3, stride=1, padding=1, bias=True)
            # 512x4x4 -> 512x4x4
            self.conv7 = EQConv2d(base * 16, out_nchan, kernel_size=3, stride=1, padding=1, bias=True)
        elif self.stage == "stage4":
            base = 512
            # 512x32x32 -> 512x32x32
            self.conv1 = EQConv2d(base, base, kernel_size=3, stride=1, padding=1, bias=True)
            # 512x16x16 -> 512x16x16
            self.conv2 = EQConv2d(base, base, kernel_size=3, stride=1, padding=1, bias=True)
            # 512x8x8 -> 512x8x8
            self.conv3 = EQConv2d(base, base, kernel_size=3, stride=1, padding=1, bias=True)
            # 512x4x4 -> 512x4x4
            self.conv4 = EQConv2d(base, base, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        if self.stage == "multiple":
            x1, x2, x3, x4 = x

            # 64x256x256 -> 64x128x128
            x1 = self.biliDown(self.act(self.conv1(x1)))
            # (64+128)x128x128 -> 64x128x128
            x2 = self.ingestion_1(torch.cat([x1, x2], dim=1))
            # 128x128x128 -> 128x64x64
            x2 = self.biliDown(self.act(self.conv2(x2)))
            # (128+256)x64x64 -> 128x64x64
            x3 = self.ingestion_2(torch.cat([x2, x3], dim=1))
            # 128x64x64 -> 128x32x32
            x3 = self.biliDown(self.act(self.conv3(x3)))
            # (128+512)x32x32 -> 128x32x32
            x4 = self.ingestion_3(torch.cat([x3, x4], dim=1))
            # 256x32x32 -> 256x16x16
            x4 = self.biliDown(self.act(self.conv4(x4)))
            # 256x16x16 -> 256x8x8
            x4 = self.biliDown(self.act(self.conv5(x4)))
            # 512x8x8 -> 512x4x4
            x4 = self.biliDown(self.act(self.conv6(x4)))
            # 512x4x4
            x4 = self.act(self.conv7(x4))
            return x4

        elif self.stage == "stage0":
            # 3x256x256
            x = self.biliDown(self.act(self.conv1(x)))
            # 32x128x128
            x = self.biliDown(self.act(self.conv2(x)))
            # 64x64x64
            x = self.biliDown(self.act(self.conv3(x)))
            # 128x32x32
            x = self.biliDown(self.act(self.conv4(x)))
            # 256x16x16
            x = self.biliDown(self.act(self.conv5(x)))
            # 512x8x8
            x = self.biliDown(self.act(self.conv6(x)))
            # 512x4x4
            x = self.act(self.conv7(x))
            # 512x4x4
            return x

        elif self.stage == "stage4":
            # 512x32x32
            x = self.biliDown(self.act(self.conv1(x)))
            # 512x16x16
            x = self.biliDown(self.act(self.conv2(x)))
            # 512x8x8
            x = self.biliDown(self.act(self.conv3(x)))
            # 512x4x4
            x = self.act(self.conv4(x))
            # 512x4x4
            return x


class Epilogue(nn.Module):
    """Class for the epilogue of the discriminator."""

    def __init__(self, stage, in_channels, res, n_cls=0, mbdis_group_size=32, mbdis_n_chan=0, cmap_dim=128):
        """Initialize discriminator epilogue.

        Args:
            stage: output stage of the feature extractor serving as input to the discriminator
            in_channels: number of input channels
            res: _description_
            n_cls: number of classes (domains). Defaults to 0.
            mbdis_group_size: group size for mini-batch discrimination. Defaults to 32.
            mbdis_n_chan: number of channels for mini-batch discrimination. Defaults to 0.
            cmap_dim: dimension of logit layer before projection layer. Defaults to 128.
        """
        super().__init__()
        self.cmap_dim = cmap_dim
        self.stage = stage
        self.mbdis = None
        self.act = nn.LeakyReLU(0.2)

        # handcrafted mbdis features
        self.mbdis = MinibatchStdLayer(mbdis_group_size, mbdis_n_chan) if mbdis_n_chan > 0 else None
        # last conv layer incorporates mbdis features
        self.conv = EQConv2d(in_channels + mbdis_n_chan, in_channels, kernel_size=3, padding=1)
        # dense layer instead of further downsampling
        self.dense = EQLinear(in_channels * (res**2), in_channels, bias=True)
        # output layer (maps to cmap_dim outputs instead of single logit)
        self.logits = EQLinear(in_channels, cmap_dim, bias=True)
        # projection layer for condition label (affine)
        self.onehot_project = EQLinear(n_cls, cmap_dim, bias=False)

    def forward(self, x, c):
        if self.mbdis is not None:
            x = self.mbdis(x)

        x = self.act(self.conv(x))
        # dense layer that does 'downsampling'
        x = self.act(self.dense(x.flatten(1)))

        logits = self.logits(x)

        # project condition
        c_proj = self.onehot_project(c)
        out = (logits * c_proj).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MinibatchStdLayer(nn.Module):
    """Mini-batch discrimination layer."""

    def __init__(self, group_size, n_chan=1):
        """Initialize mini-batch discrimination layer.

        Args:
            group_size: group size
            n_chan: number of channels. Defaults to 1.
        """
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        N, C, H, W = x.shape
        G = N
        if self.group_size is not None:
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
        F = self.n_chan
        c = C // F

        # split minibatch in n groups of size G, split channels in F groups of size c
        y = x.reshape(G, -1, F, c, H, W)
        # shift center (per group) to zero
        y = y - y.mean(dim=0)
        # variance per group
        y = y.square().mean(dim=0)
        # stddev
        y = (y + 1e-8).sqrt()
        # average over channels and pixels
        y = y.mean(dim=[2, 3, 4])
        # reshape and tile
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        # add to input as 'handcrafted feature channels'
        x = torch.cat([x, y], dim=1)
        return x


class CondEqlDiscriminator(nn.Module):
    """Conditional discriminator."""

    def __init__(self, feat_matching_layer, device, n_cls=0, mbdis=True):
        """Initialize conditional discriminator.

        Args:
            feat_matching_layer: feature stage/layer serving as input to the discriminator
            device: gpu or cpu device
            n_cls: number of classes (domains). Defaults to 0.
            mbdis: whether to apply mini-batch discrimination. Defaults to True.
        """
        super(CondEqlDiscriminator, self).__init__()
        self.device = device
        # prologue, depending on matching stage
        print("Matching stage {}".format(feat_matching_layer))
        # interface between prologue and epilogue
        epilogue_in_res = 4
        epilogue_in_nchan = 512
        if feat_matching_layer == "multiple":
            stage = "multiple"
        elif feat_matching_layer == "stage0":
            stage = "stage0"
        elif feat_matching_layer == "stage4":
            stage = "stage4"
        self.prologue = Prologue(stage=stage, out_nchan=epilogue_in_nchan)
        # epilogue, with optional mbdis
        mbdis_n_chan = 1 if mbdis else 0
        self.logits = Epilogue(stage, epilogue_in_nchan, n_cls=n_cls, res=epilogue_in_res, mbdis_n_chan=mbdis_n_chan)
        self.to(device)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x, c):
        x = self.prologue(x)
        x = self.logits(x, c)
        return x
