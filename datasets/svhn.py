import pathlib
from typing import Tuple

import torchvision
from torch.utils.data.dataset import Dataset

import datasets.registry as registry
import datasets.utils as dataset_utils


@registry.register("svhn")
def load_svhn(data_dir: pathlib.Path) -> Tuple[Dataset, Dataset]:
    """Loads and returns train and test datasets for SVHN."""
    train_transform, test_transform = dataset_utils.compose_transforms(
        shape=(32, 32, 3),
        normalize_mean=(0.4377, 0.4438, 0.4728),
        normalize_std=(0.1980, 0.2010, 0.1970),
    )
    svhn_datadir = data_dir / "svhn"
    train = torchvision.datasets.SVHN(
        svhn_datadir, split="train", transform=train_transform, download=True
    )
    test = torchvision.datasets.SVHN(
        svhn_datadir, split="test", transform=test_transform, download=True
    )

    return train, test
