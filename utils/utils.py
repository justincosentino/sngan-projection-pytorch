import json
import pathlib
import shutil
from typing import Any
from typing import Dict
from typing import List

import torch


def build_experiment_name(args: Dict[str, Any], keys: List[str] = None) -> str:
    """
    Builds an experiment id string given a dictionary of command line arguments.

    Args:
        args: a dictionary of commnand line arguments.
        keys: a list of keys to include in tthe experiment id (ordered).

    Returns:
        A string representation of the args dictionary.
    """
    if keys is None:
        keys = [
            "dataset",
            "epochs",
            "lr",
            "batch_size",
            "alpha",
            "target_coverage",
            "backbone",
            "no_aux_head",
        ]
    str_rep: List[str] = []
    for key in keys:
        value = args[key]
        str_rep.append(f"{key}={value}")
    return "&".join(str_rep)


def write_dict(args: Dict[str, Any], path: pathlib.Path) -> None:
    """Writes a dictionary to the filesystem as a json blog."""
    with open(path, "wt") as f:
        f.write(json.dumps(args))


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    path: pathlib.Path,
    cktp_name="cktp.pth.tar",
    best_cktp_name="best_cktp.pth.tar",
) -> None:
    """Saves a state dict checkpoint. If specified, copies the best checkpoint."""
    file_path = path / cktp_name
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, path / best_cktp_name)
