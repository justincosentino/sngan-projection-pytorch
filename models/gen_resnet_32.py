import torch

import models.registry as registry
from gen_resblock import GenBlock


@registry.register("generator_resnet_32")
class GenResNet32(torch.nn.Module):
    def __init__(
        self,
        num_channels=256,
        dim_z=128,
        bottom_width=4,
        activation=torch.nn.functional.relu,
        num_classes=0,
        distribution="normal",
    ):
        super(GenResNet32, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.num_classes = num_classes

        self.l1 = torch.nn.Linear(
            in_features=dim_z, out_features=((bottom_width ** 2) * num_channels)
        )
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=1)
        torch.nn.init.zeros_(self.l1.bias)

        self.block2 = GenBlock(
            in_channels=(num_channels),
            out_channels=(num_channels),
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = GenBlock(
            in_channels=(num_channels),
            out_channels=(num_channels),
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = GenBlock(
            in_channels=(num_channels),
            out_channels=(num_channels),
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )

        self.b5 = torch.nn.BatchNormalization(num_channels)
        self.c5 = torch.nn.Convolution2D(
            in_channels=num_channels, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.xavier_uniform_(self.c5.weight, gain=1)
        torch.nn.init.zeros_(self.c5.bias)

    def forward(self, batchsize=64, y=None, z=None, **kwargs):
        # if z is None:
        #     z = sample_continuous(
        #         self.dim_z, batchsize, distribution=self.distribution, xp=self.xp
        #     )
        # if y is None:
        #     y = (
        #         sample_categorical(
        #             self.num_classes, batchsize, distribution="uniform", xp=self.xp
        #         )
        #         if self.num_classes > 0
        #         else None
        #     )
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise ValueError(f"z.shape[0] != y.shape[0], {z.shape[0]} != {y.shape[0]}")
        h = z
        h = self.l1(h)
        h = h.view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))
        return h
