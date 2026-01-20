#optim_utils.py
from __future__ import annotations
import torch
import torch.nn as nn
from model_utils import get_param_groups, set_backbone_bn_eval, freeze_backbone, unfreeze_last_blocks

def build_optimizer(model: nn.Module, backbone_lr_factor: float = 0.1,
                    lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Build Adam optimizer with separate LR for head and backbone."""
    param_groups, _, _ = get_param_groups(model, backbone_lr_factor, lr, weight_decay)
    return torch.optim.Adam(param_groups)

def build_plateau_scheduler(optimizer: torch.optim.Optimizer, mode: str = "max",
                            factor: float = 0.5, patience: int = 2, verbose: bool = True):
    """ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
    )

__all__ = [
    "set_backbone_bn_eval",
    "freeze_backbone",
    "unfreeze_last_blocks",
    "get_param_groups",
    "build_optimizer",
    "build_plateau_scheduler"
]
