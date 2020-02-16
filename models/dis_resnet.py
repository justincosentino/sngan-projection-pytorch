import torch

import models.registry as registry
from models.dis_resblock import ResBlock
from models.dis_resblock import OptimizedBlock


@registry.register("sn_projection_discriminator_resnet")
class SNProjectionDisResNet(torch.nn.Module):
    def __init__(
        self, num_channels=64, num_classes=0, activation=torch.nn.functional.relu
    ):
        super(SNProjectionDisResNet, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(in_channels=3, out_channels=num_channels)

        self.block2 = ResBlock(
            in_channels=num_channels,
            out_channels=(num_channels * 2),
            activation=activation,
            downsample=True,
        )

        self.block3 = ResBlock(
            in_channels=(num_channels * 2),
            out_channels=(num_channels * 4),
            activation=activation,
            downsample=True,
        )

        self.block4 = ResBlock(
            in_channels=(num_channels * 4),
            out_channels=(num_channels * 8),
            activation=activation,
            downsample=True,
        )

        self.block5 = ResBlock(
            in_channels=(num_channels * 8),
            out_channels=(num_channels * 16),
            activation=activation,
            downsample=True,
        )

        self.block6 = ResBlock(
            in_channels=(num_channels * 16),
            out_channels=(num_channels * 16),
            activation=activation,
            downsample=False,
        )

        self.l7 = torch.nn.utils.spectral_norm(
            torch.nn.Linear(in_features=(num_channels * 16), out_features=1)
        )
        torch.nn.init.xavier_uniform_(self.l7.weight, gain=1)
        torch.nn.init.zeros_(self.l7.bias)

        if num_classes > 0:
            self.l_y = torch.nn.utils.spectral_norm(
                torch.nn.Embedding(
                    num_embeddings=num_classes, embedding_dim=num_channels * 16
                )
            )
            torch.nn.init.xavier_uniform_(self.l_y.weight, gain=1)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, axis=(2, 3))
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y * h, axis=1, keepdims=True)
        return output


@registry.register("sn_concat_discriminator_resnet")
class SNConcatDisResNet(torch.nn.Module):
    def __init__(
        self,
        num_channels=64,
        num_classes=0,
        activation=torch.nn.functional.relu,
        dim_emb=128,
    ):
        super(SNProjectionDisResNet, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(in_channels=3, out_channels=num_channels)

        self.block2 = ResBlock(
            in_channels=num_channels,
            out_channels=(num_channels * 2),
            activation=activation,
            downsample=True,
        )

        self.block3 = ResBlock(
            in_channels=(num_channels * 2),
            out_channels=(num_channels * 4),
            activation=activation,
            downsample=True,
        )

        self.l_y = torch.nn.utils.spectral_norm(
            torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=dim_emb)
        )
        torch.nn.init.xavier_uniform_(self.l_y.weight, gain=1)

        self.block4 = ResBlock(
            in_channels=(num_channels * 4 + dim_emb),
            out_channels=(num_channels * 8),
            activation=activation,
            downsample=True,
        )

        self.block5 = ResBlock(
            in_channels=(num_channels * 8),
            out_channels=(num_channels * 16),
            activation=activation,
            downsample=True,
        )

        self.block6 = ResBlock(
            in_channels=(num_channels * 16),
            out_channels=(num_channels * 16),
            activation=activation,
            downsample=False,
        )

        self.l7 = torch.nn.utils.spectral_norm(
            torch.nn.Linear(in_features=(num_channels * 16), out_features=1)
        )
        torch.nn.init.xavier_uniform_(self.l7.weight, gain=1)
        torch.nn.init.zeros_(self.l7.bias)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, axis=(2, 3))
        output = self.l7(h)
        return output
