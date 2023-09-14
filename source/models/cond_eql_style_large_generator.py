import torch
from torch import nn

from .layers import StyleEQConv2dWithBias, EQLinear


class CondEqlGenerator(nn.Module):
    """Conditional generator."""

    def __init__(self, latent_dim, device, n_cls=0, out_channels=1):
        """Initialize conditional generator.

        Args:
            latent_dim: latent noise vector dimension
            device: gpu or cpu device
            n_cls: number of classes (domains). Defaults to 0.
            out_channels: number of output channels. Defaults to 1.
        """
        super(CondEqlGenerator, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        # dimension of linear one-hot embedding
        onehot_embed_d = 256
        # activation used whenever non-linear
        self.act = nn.LeakyReLU(0.2)
        # bilinear upsampling layer
        self.biliUp = nn.UpsamplingBilinear2d(scale_factor=2)

        # embed one-hot
        self.onehot_embed = EQLinear(n_cls, onehot_embed_d, bias=True)
        # map latent + one-hot to w
        self.w1 = EQLinear(latent_dim + onehot_embed_d, latent_dim, bias=True, lr_mul=0.01)
        self.w2 = EQLinear(latent_dim, latent_dim, bias=True, lr_mul=0.01)

        # number of input channels for each layer
        in_nchans = [None, 512, 512, 256, 256, 128, 128, 64, 64, 64, 64]
        base_res = 4
        base_nchan = 512
        # learned constant
        self.const = nn.Parameter(torch.ones((1, base_nchan, base_res, base_res)))
        # conv layers, styles, and noise scales
        for i in range(1, 10):
            in_nchan = in_nchans[i]
            out_nchan = in_nchans[i + 1]
            conv = StyleEQConv2dWithBias(
                in_nchan, out_nchan, kernel_size=3, stride=1, padding=1, bias=True, device=device
            )
            setattr(self, "conv{}".format(i), conv)
        # output layer (no noise)
        self.out_layer = StyleEQConv2dWithBias(
            in_nchans[-1], out_channels, kernel_size=3, stride=1, padding=1, bias=True, noise=False
        )

        self.to(device)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x, noise=None):
        # split in z, one-hot
        z = x[:, : self.latent_dim]
        one_hot = x[:, self.latent_dim :]

        # embed (linearly), normalize, and concatenate
        one_hot = self.onehot_embed(one_hot)
        x = torch.cat([z, one_hot], dim=1)

        # map to w
        w = self.act(self.w1(x))
        w = self.act(self.w2(w))

        # broadcast learned constant along batch dim
        bs = x.size()[0]
        x = self.const.expand([bs, -1, -1, -1])

        # style convolutions
        for i in range(1, 10):
            style_conv = getattr(self, "conv{}".format(i))
            n = noise[i] if noise is not None else None
            x = style_conv((x, w), n=n)
            x = self.act(x)
            if i in [2, 3, 4, 5, 6, 8]:
                x = self.biliUp(x)
            # 512x4x4 -> 512x4x4
            # 512x4x4 -> 256x4x4 -> 256x8x8
            # 256x8x8 -> 256x8x8 -> 256x16x16
            # 256x16x16 -> 128x16x16 -> 128x32x32
            # 128x32x32 -> 128x32x32 -> 128x64x64
            # 128x64x64 -> 64x64x64 -> 64x128x128
            # 64x128x128 -> 64x128x128
            # 64x128x128 -> 64x128x128 -> 64x256x256
            # 64x256x256 -> 64x256x256
            # 64x256x256 -> 3x256x256

        # linear output
        return self.out_layer((x, w))
