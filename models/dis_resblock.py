import torch


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        kernel_size=3,
        padding=1,
        activation=torch.nn.functional.relu,
        downsample=False,
    ):

        super(ResBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.c1 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c1.bias)

        self.c2 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c2.bias)

        if self.learnable_sc:
            self.c_sc = torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            torch.nn.init.xavier_uniform_(self.c_sc.weight, gain=1)
            torch.nn.init.zeros_(self.c_sc.bias)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = torch.nn.functional.avg_pool2d(h, kernel_size=(2, 2))
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2))
            return x
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        activation=torch.nn.functional.relu,
    ):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c1.bias)

        self.c2 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2 ** 0.5))
        torch.nn.init.zeros_(self.c2.bias)

        self.c_sc = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        torch.nn.init.xavier_uniform_(self.c_sc.weight, gain=1)
        torch.nn.init.zeros_(self.c_sc.bias)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = torch.nn.functional.avg_pool2d(h, kernel_size=(2, 2))
        return h

    def shortcut(self, x):
        return self.c_sc(torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2)))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
