from torch import nn
import torchvision


class Encoder(nn.Module):
    """VGG encoder."""

    def __init__(self):
        """Initialize VGG encoder."""
        super(Encoder, self).__init__()

        # create vgg19 instance
        vgg19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg19.parameters():
            param.requires_grad = False

        self.noutchannels = vgg19.features[(19)].out_channels

        # replace max-pooling with identity and increase stride of the preceding conv layer
        for n, m in vgg19.features.named_children():
            if isinstance(m, nn.MaxPool2d):
                vgg19.features[int(n)] = nn.Identity()
                vgg19.features[int(n) - 2].stride = (2, 2)

        self.layers = nn.Sequential()
        self.layers1 = nn.Sequential()
        self.layers2 = nn.Sequential()
        self.layers3 = nn.Sequential()
        self.layers4 = nn.Sequential()
        for _i in range(21):
            self.layers.add_module("module_{}".format(_i), vgg19.features[(_i)])
            if _i <= 1:
                self.layers1.add_module("module_{}".format(_i), vgg19.features[(_i)])
            if 2 <= _i <= 6:
                self.layers2.add_module("module_{}".format(_i), vgg19.features[(_i)])
            if 7 <= _i <= 11:
                self.layers3.add_module("module_{}".format(_i), vgg19.features[(_i)])
            if 12 <= _i <= 20:
                self.layers4.add_module("module_{}".format(_i), vgg19.features[(_i)])

    def forward(self, x, multiple=False):
        if multiple:
            assert len(x.shape) == 4
            x1 = self.layers1(x)
            x2 = self.layers2(x1)
            x3 = self.layers3(x2)
            x4 = self.layers4(x3)
            return x1, x2, x3, x4
        else:
            assert len(x.shape) == 4
            x = self.layers(x)
            return x
