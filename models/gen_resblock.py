import torch


def upsample_conv(x, conv):
    h, w = x.size()[2:]
    x_upsample = torch.nn.functional.interpolate(x, size=(h * 2, w * 2), mode="nearest")
    return conv(x_upsample)


class GenBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        kernel_size=3,
        padding=1,
        activation=torch.nn.functional.relu,
        upsample=False,
        num_classes=0,
    ):

        super(GenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.num_classes = num_classes

        self.c1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c1.bias)

        self.c2 = torch.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c2.bias)

        if num_classes > 0:
            self.b1 = ConditionalBatchNorm2d(
                num_features=in_channels, num_classes=num_classes
            )
            self.b2 = ConditionalBatchNorm2d(
                num_features=hidden_channels, num_classes=num_classes
            )
        else:
            self.b1 = torch.nn.BatchNorm2d(num_features=in_channels, eps=2e-5)
            self.b2 = torch.nn.BatchNorm2d(num_features=hidden_channels, eps=2e-5)

        if self.learnable_sc:
            self.c_sc = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            torch.nn.init.xavier_uniform_(self.c_sc.weight, gain=1)
            torch.nn.init.zeros_(self.c_sc.bias)

    def residual(self, x, y=None, z=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        return x

    def forward(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)


# TODO: compare this to chainer
class ConditionalBatchNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm2d(num_features, affine=False)
        self.embed = torch.nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise bias at 0
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out
