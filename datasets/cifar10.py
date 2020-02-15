import pathlib
from typing import Tuple

import torchvision
from torch.utils.data.dataset import Dataset

import datasets.registry as registry
import datasets.utils as dataset_utils


@registry.register("cifar10")
def load_cifar10(data_dir: pathlib.Path) -> Tuple[Dataset, Dataset]:
    """Loads and returns train and test datasets for CIFAR10."""
    train_transform, test_transform = dataset_utils.compose_transforms(
        shape=(32, 32, 3),
        normalize_mean=(0.4914, 0.4822, 0.4465),
        normalize_std=(0.2023, 0.1994, 0.2010),
    )
    cifar10_data_dir = data_dir / "cifar10"
    train = torchvision.datasets.CIFAR10(
        cifar10_data_dir, train=True, transform=train_transform, download=True
    )
    test = torchvision.datasets.CIFAR10(
        cifar10_data_dir, train=False, transform=test_transform, download=True
    )

    return train, test
