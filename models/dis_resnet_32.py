import torch

import models.registry as registry
from models.dis_resblock import ResBlock
from models.dis_resblock import OptimizedBlock


@registry.register("sn_projection_discriminator_resnet_32")
class SNProjectionDisResNet32(torch.nn.Module):
    def __init__(
        self, num_channels=128, num_classes=0, activation=torch.nn.functional.relu
    ):
        super(SNProjectionDisResNet32, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(in_channels=3, out_channels=num_channels)

        self.block2 = ResBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            activation=activation,
            downsample=True,
        )

        self.block3 = ResBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            activation=activation,
            downsample=False,
        )

        self.block4 = ResBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            activation=activation,
            downsample=False,
        )

        self.l5 = torch.nn.utils.spectral_norm(
            torch.nn.Linear(in_features=num_channels, out_features=1)
        )
        torch.nn.init.xavier_uniform_(self.l5.weight, gain=1)
        torch.nn.init.zeros_(self.l5.bias)

        if num_classes > 0:
            self.l_y = torch.nn.utils.spectral_norm(
                torch.nn.Embedding(
                    num_embeddings=num_classes, embedding_dim=num_channels
                )
            )
            torch.nn.init.xavier_uniform_(self.l_y.weight, gain=1)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, axis=(2, 3))
        output = self.l5(h)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y * h, axis=1, keepdims=True)
        return output
