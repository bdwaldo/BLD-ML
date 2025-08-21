# src/model_utils.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

def set_backbone_bn_eval(m: nn.Module):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def freeze_backbone(model: nn.Module, freeze_bn: bool = True):
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False
    if freeze_bn and hasattr(model, "features"):
        model.features.apply(set_backbone_bn_eval)

def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2, freeze_bn: bool = True):
    if not hasattr(model, "features"):
        return
    # EfficientNetV2.features is a Sequential of blocks
    blocks = list(model.features.children())[-max(1, n_blocks):]
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
        if freeze_bn:
            b.apply(set_backbone_bn_eval)
