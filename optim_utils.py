# src/optim_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import torch
import torch.nn as nn

def get_param_groups(model: nn.Module, backbone_lr_factor: float, base_lr: float, weight_decay: float):
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": base_lr * backbone_lr_factor, "weight_decay": weight_decay},
    ]
    return param_groups, head_params, backbone_params


def build_optimizer(model: nn.Module, backbone_lr_factor: float = 0.1,
                    lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    param_groups, _, _ = get_param_groups(model, backbone_lr_factor, lr, weight_decay)
    return torch.optim.Adam(param_groups)


def build_plateau_scheduler(optimizer: torch.optim.Optimizer, mode: str = "max",
                            factor: float = 0.5, patience: int = 2, verbose: bool = True):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
    )
