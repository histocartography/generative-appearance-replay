import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Linear


class EQConv2d(Conv2d):
    """Convolution layer with equalized learning rate."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        """Initialize convolutional layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size
            stride: stride. Defaults to 1.
            padding: padding. Defaults to 0.
            dilation: dilation. Defaults to 1.
            groups: groups. Defaults to 1.
            bias: bias. Defaults to True.
            padding_mode: padding mode. Defaults to "zeros".
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # initialize weights from normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # equalized lr
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class StyleEQConv2dWithBias(EQConv2d):
    """Style convolution with equalized learning rate."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        wdim=512,
        stylemod=True,
        noise=True,
        device=None,
    ):
        """Initialize style convolution.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size
            stride: stride. Defaults to 1.
            padding: padding. Defaults to 0.
            dilation: dilation. Defaults to 1.
            groups: groups. Defaults to 1.
            bias: bias. Defaults to True.
            padding_mode: padding mode. Defaults to "zeros".
            wdim: dimension of w. Defaults to 512.
            stylemod: style modulation. Defaults to True.
            noise: whether to add noise. Defaults to True.
            device: gpu or cpu device. Defaults to None.
        """
        super(StyleEQConv2dWithBias, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.device = device
        self.out_channels = out_channels
        # no bias, scale only
        self.y1 = EQLinear(wdim, in_channels, bias_init=1.0) if stylemod else None
        # bias
        self.y2 = EQLinear(wdim, in_channels, bias_init=0.0) if stylemod else None
        # single noise scalar
        self.noise_scale = nn.Parameter(torch.zeros([1, out_channels, 1, 1])) if noise else None

    def forward(self, x, n=None):
        x, w = x
        bs, nchan, res = x.size()[:3]
        # style modulation
        if self.y1 is not None and self.y2 is not None:
            y1 = self.y1(w)
            y2 = self.y2(w)
            y1 = y1.reshape(-1, nchan, 1, 1)
            y2 = y2.reshape(-1, nchan, 1, 1)
            x = x * y1 + y2

        # convolution
        x = super(StyleEQConv2dWithBias, self).forward(x)
        # add noise
        if self.noise_scale is not None:
            n = torch.randn((bs, 1, res, res), device=self.device) if n is None else n
            x += self.noise_scale * n
        return x


class EQLinear(Linear):
    """Linear layer with equalized learning rate."""

    def __init__(self, in_features, out_features, bias=True, bias_init=0.0, lr_mul=1.0):
        """Initialize linear layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            bias: bias. Defaults to True.
            bias_init: bias intialization. Defaults to 0..
            lr_mul: learning rate factor. Defaults to 1..
        """
        super().__init__(in_features, out_features, bias)

        # initialize weights from normal distribution
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0 / lr_mul)

        # initialize bias
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None

        # equalized lr
        fan_in = self.in_features
        self.weight_scale = lr_mul / np.sqrt(fan_in)
        self.bias_scale = lr_mul

    def forward(self, x):
        w = self.weight * self.weight_scale
        if self.bias is not None and self.bias_scale != 1:
            b = self.bias * self.bias_scale
        else:
            b = self.bias
        return torch.nn.functional.linear(x, w, b)


class AdaIN(nn.Module):
    """Adaptive instance normalization (AdaIN)."""

    def __init__(self, noutchannels, disabled=False):
        """Initialize AdaIN.

        Args:
            noutchannels: number of output channels
            disabled: whether to disable the layer. Defaults to False.
        """
        super(AdaIN, self).__init__()
        self.disabled = disabled
        self.noutchannels = noutchannels

    def forward(self, contentf, stylef):
        if self.disabled:
            self.output = contentf
            return self.output
        N, c, x, y = contentf.shape

        # contentView.shape = (N, c, x*y)
        contentView = contentf.view(*contentf.shape[:2], -1)
        # contentMean.shape = (N, c)
        contentMean = contentView.mean(-1)
        # contentCentred.shape = (N, c, x*y)
        contentCentered = contentView - contentMean.view(N, c, 1)
        # contentStd.shape = (N, c)
        contentStd = ((contentCentered**2).mean(-1) + 1e-6).sqrt()
        # contentMean.shape = (N, c, 1, 1)
        contentMean = contentMean.view(*contentMean.shape[:2], *((len(contentf.shape) - 2) * [1]))
        # contentStd.shape = (N, c, 1, 1)
        contentStd = contentStd.view(*contentStd.shape[:2], *((len(contentf.shape) - 2) * [1]))

        # styleView.shape = (N, c, x*y)
        styleView = stylef.view(*stylef.shape[:2], -1)
        # styleMean.shape = (N, c)
        styleMean = styleView.mean(-1)
        # styleCentred.shape = (N, c, x*y)
        styleCentered = styleView - styleMean.view(N, c, 1)
        # styleStd.shape = (N, c)
        styleStd = ((styleCentered**2).mean(-1) + 1e-6).sqrt()
        # styleMean.shape = (N, c, 1, 1)
        styleMean = styleMean.view(*styleMean.shape[:2], *((len(stylef.shape) - 2) * [1]))
        # styleStd.shape = (N, c, 1, 1)
        styleStd = styleStd.view(*styleStd.shape[:2], *((len(stylef.shape) - 2) * [1]))

        out = ((contentf - contentMean) / contentStd) * styleStd + styleMean

        return out
