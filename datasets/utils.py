from typing import Any
from typing import Tuple

import torchvision


def compose_transforms(
    shape: Tuple[int, int, int],
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    random_crop_padding: int = 4,
    random_rotate_degrees: Tuple[float, float] = (-15, 15),
) -> Tuple[Any, Any]:
    """
    Generates a composition of torchvision.transforms for training and test data.

    Training transforms include:
        - random crop
        - random rotation
        - random horizontal flip
        - conversion to tensors
        - mean and std normalization

    Test transforms include:
        - conversion to tensors
        - mean and std normalization

    Args:
        shape: the target image shape.
        normalize_mean: channel-wise mean across training images.
        normalize_std: channel-wise std across training images.
        random_crop_padding: size of padding used in random crop.
        random_rotate_degrees: range of degrees to select from.

    Returns:
        Composed transforms for the train and test data.

    Raises:
        None.
    """
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(shape[:2], padding=random_crop_padding),
            torchvision.transforms.RandomRotation(degrees=random_rotate_degrees),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(normalize_mean, normalize_std),
        ]
    )
    return train_transform, test_transform
