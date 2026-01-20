#model_utils.py
from __future__ import annotations
import torch.nn as nn

def set_backbone_bn_eval(m: nn.Module):
    """Set BatchNorm layers to eval mode."""
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def get_head_names(model: nn.Module):
    """Return the attribute names of the classifier head."""
    if hasattr(model, "classifier"):
        return ["classifier"]
    elif hasattr(model, "fc"):
        return ["fc"]
    else:
        # Fallback: assume last child is the head
        return [list(dict(model.named_children()).keys())[-1]]

def freeze_backbone(model: nn.Module, freeze_bn: bool = True):
    """Freeze all backbone parameters except the head."""
    head_names = get_head_names(model)
    for name, p in model.named_parameters():
        if not any(hn in name for hn in head_names):
            p.requires_grad = False
    if freeze_bn:
        if hasattr(model, "features"):  # EfficientNet / MobileNet
            model.features.apply(set_backbone_bn_eval)
        elif hasattr(model, "layer1"):  # ResNet
            for name, module in model.named_children():
                if name not in head_names:
                    module.apply(set_backbone_bn_eval)
        elif hasattr(model, "Mixed_7c"):  # Inception
            for name, module in model.named_children():
                if name not in head_names:
                    module.apply(set_backbone_bn_eval)

def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2, freeze_bn: bool = True):
    """Unfreeze the last n_blocks of the backbone."""
    blocks = []

    if hasattr(model, "features"):  # EfficientNet / MobileNet
        blocks = list(model.features.children())[-max(1, n_blocks):]

    elif hasattr(model, "layer4"):  # ResNet
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for layer in layers:
            if len(blocks) >= n_blocks:
                break
            blocks.extend(reversed(list(layer.children())))
        blocks = list(reversed(blocks[:n_blocks]))

    elif hasattr(model, "Mixed_7c"):  # Inception
        blocks = [model.Mixed_7c]  # treat as one block

    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
        if freeze_bn:
            b.apply(set_backbone_bn_eval)

def get_param_groups(model: nn.Module, backbone_lr_factor: float, base_lr: float, weight_decay: float):
    """Separate head and backbone params for differential learning rates."""
    head_names = get_head_names(model)
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(hn in name for hn in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": base_lr * backbone_lr_factor, "weight_decay": weight_decay},
    ], head_params, backbone_params
