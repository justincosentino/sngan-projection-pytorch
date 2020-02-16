import torch
import torchvision

import models.registry as registry


@registry.register("vgg")
class VggSelectiveNet(torch.nn.Model):
    """
    A VGG-based [1] implementation of SelectiveNet [2].

    [1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
    large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

    [2] Geifman, Yonatan, and Ran El-Yaniv. "Selectivenet: A deep neural network with an
    integrated reject option." arXiv preprint arXiv:1901.09192 (2019).
    """

    def __init__(self, num_classes=10, use_auxiliary_head=True):
        super(VggSelectiveNet, self).__init__()

        self.use_auxiliary_head = use_auxiliary_head

        # Init VGG backbone.
        vgg = torchvision.models.vgg16_bn(
            pretrained=False, progress=True, num_classes=num_classes
        )
        vgg.classifier = vgg.classier[:-1]
        self.backbone = vgg

        # Init classificatiton head, f().
        self.cls = torch.nn.Linear(
            in_features=4096, out_features=num_classes, bias=True
        )

        # Init selection head, g().
        self.sel = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=512, bias=True),
            torch.nn.BatchNorm1d(
                num_features=512,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            torch.nn.ReLU(inplace=True),
            # TODO: consider adding a lambda x:x/10 layer as done in SN impl.
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
            torch.nn.Sigmoid(),
        )

        # Init auxiliary head, h().
        if use_auxiliary_head:
            self.aux = torch.nn.Linear(
                in_features=4096, out_features=num_classes, bias=True
            )

    def forward(self, x):
        features = self.backbone(x)
        f_out = self.cls(features)
        g_out = self.sel(features)

        if self.use_auxiliary_head and self.aux is not None:
            return f_out, g_out, self.aux(features)

        return f_out, g_out, f_out
