# src/__init__.py
"""
Nematode ML package
Provides training loop, optimizer utilities, and model helpers.
"""

from .training import train_1_binary
from .optim_utils import build_optimizer, build_plateau_scheduler
from .model_utils import (
    freeze_backbone,
    unfreeze_last_blocks,
    get_param_groups,
    set_backbone_bn_eval,
)

__all__ = [
    "train_1_binary",
    "build_optimizer",
    "build_plateau_scheduler",
    "freeze_backbone",
    "unfreeze_last_blocks",
    "get_param_groups",
    "set_backbone_bn_eval",
]
