"""Basic registry for model builders."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import Text

import torch

_MODELS = dict()


def register(model_name: Text) -> Callable:
    """Registers a new model builder under the given model name."""

    def add_to_dict(func: Callable) -> Callable:
        _MODELS[model_name] = func
        return func

    return add_to_dict


def load_model(model_name: Text, params: Dict[str, Any]) -> torch.nn.Module:
    """Fetches and invokes the model builder associated with the given model name.

    Args:
        model_name: The registered name of the model loader.
        params: a dict of kwargs passed to the dataset loader.

    Returns:
        A model.

    Raises:
        ValueError: if an unknown model name is specified.
    """
    if model_name not in _MODELS:
        raise Exception("Invalid model builder: {}".format(model_name))
    return _MODELS[model_name](**params)
