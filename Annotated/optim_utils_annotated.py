#optim_utils.py

#defines get_param_groups needed for trainin.py to build optimizers with differential learning rates

from __future__ import annotations #__future__.annotations: allows postponed evaluation of type hints (so you can use forward references without quotes).

import torch
import torch.nn as nn
from model_utils import get_param_groups, set_backbone_bn_eval, freeze_backbone, unfreeze_last_blocks
#get_param_groups splits parameters into groups (e.g., backbone vs classifier) with different learning rates.
#set_backbone_bn_eval sets BatchNorm layers to eval mode (common in transfer learning).
#freeze_backbone freezes backbone weights
#unfreeze_last_blocks: progressively unfreezes layers for fine‑tuning

#Create an Adam optimizer with parameter groups
def build_optimizer(model: nn.Module, backbone_lr_factor: float = 0.1, #Backbone parameters get lr * backbone_lr_factor (e.g., 0.1× base LR)
                    lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Build Adam optimizer with separate LR for head and backbone."""
    param_groups, _, _ = get_param_groups(model, backbone_lr_factor, lr, weight_decay) #get_param_groups: returns parameter groups with adjusted learning rates
    #Head/classifier parameters get full lr
    #weight_decay applied for regularization
    return torch.optim.Adam(param_groups)

#Wraps PyTorch’s ReduceLROnPlateau scheduler
#"min": reduce LR when monitored metric stops decreasing (e.g., validation loss).
#"max": reduce LR when monitored metric stops increasing (e.g., accuracy, AUROC).
#Factor: multiply LR by this factor when triggered (e.g., 0.5 halves LR).
#Patience: number of epochs with no improvement before reducing LR.
#Verbose: prints LR changes.
def build_plateau_scheduler(optimizer: torch.optim.Optimizer, mode: str = "max",
                            factor: float = 0.5, patience: int = 2, verbose: bool = True):
    """ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
    )

#Defines the public API of this module
#When you from optim_utils import *, only these names are imported
#Re‑exports backbone utilities from model_utils plus the optimizer/scheduler builders
__all__ = [
    "set_backbone_bn_eval",
    "freeze_backbone",
    "unfreeze_last_blocks",
    "get_param_groups",
    "build_optimizer",
    "build_plateau_scheduler"
]
