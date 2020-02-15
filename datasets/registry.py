"""Basic registry for dataset loaders."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import Text
from typing import Tuple

from torch.utils.data.dataset import Dataset

_DATASETS = dict()


def register(dataset: Text) -> Callable:
    """Registers a new dataset loader under the given dataset name."""

    def add_to_dict(func: Callable) -> Callable:
        _DATASETS[dataset] = func
        return func

    return add_to_dict


def load_dataset(dataset: Text, params: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Fetches and invokes the dataset loader associated with the given dataset name.

    Args:
        dataset: the registered name of the dataset loader.
        params: a dict of kwargs passed to the dataset loader.

    Returns:
        The train and test datasets.

    Raises:
        ValueError: if an unknown dataset name is specified.
    """
    if dataset not in _DATASETS:
        raise Exception("Invalid dataset loader: {}".format(dataset))
    return _DATASETS[dataset](**params)
