#model_utils.py

#foundation of the transfer learning strategy. 
#Identifies the classifier head, freezes/unfreezes backbone layers, stabilizes BatchNorm, and separates parameters for differential learning rates.

from __future__ import annotations #postpones evaluation of type hints (cleaner forward references).
import torch.nn as nn

def set_backbone_bn_eval(m: nn.Module):
    """Set BatchNorm layers to eval mode. 
    Prevents BatchNorm statistics from drifting when backbone is frozen."""
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

#Identify the classifier head
def get_head_names(model: nn.Module):
    """Return the attribute names of the classifier head."""
    if hasattr(model, "classifier"):
        return ["classifier"] #EfficientNet/MobileNet
    elif hasattr(model, "fc"): #ResNet
        return ["fc"]
    else:
        # Fallback: assume last child is the head
        #Returns a list of head names so you can distinguish head vs backbone parameters
        return [list(dict(model.named_children()).keys())[-1]] 

#Freeze backbone
def freeze_backbone(model: nn.Module, freeze_bn: bool = True):
    """Freeze all backbone parameters except the head."""
    head_names = get_head_names(model)
    for name, p in model.named_parameters():
        if not any(hn in name for hn in head_names):
            p.requires_grad = False #Sets requires_grad=False for backbone parameters so they won’t be updated
    if freeze_bn: #Leaves head parameters trainable
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

#Progressive unfreezing
def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2, freeze_bn: bool = True):
    """Unfreeze the last n_blocks of the backbone for fine-tuning.
    Different logic depending on model type"""
    blocks = []

    if hasattr(model, "features"):  # EfficientNet / MobileNet
        blocks = list(model.features.children())[-max(1, n_blocks):] #last children of features

    elif hasattr(model, "layer4"):  # ResNet
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for layer in layers:
            if len(blocks) >= n_blocks: #progressively unfreezes from layer4 backwards
                break
            blocks.extend(reversed(list(layer.children())))
        blocks = list(reversed(blocks[:n_blocks]))

    elif hasattr(model, "Mixed_7c"):  # Inception --> unfreezes Mixed_7c
        blocks = [model.Mixed_7c]  # treat as one block

    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
        if freeze_bn: #Keeps BatchNorm frozen if freeze_bn=True
            b.apply(set_backbone_bn_eval)

#Parameter grouping
def get_param_groups(model: nn.Module, backbone_lr_factor: float, base_lr: float, weight_decay: float):
    """Separate head and backbone params into groups for differential learning rates."""
    head_names = get_head_names(model) #Head: full learning rate (base_lr).
    head_params, backbone_params = [], [] 
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(hn in name for hn in head_names):
            head_params.append(p) 
        else:
            backbone_params.append(p) #Backbone: scaled learning rate (base_lr * backbone_lr_factor).
    return [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": base_lr * backbone_lr_factor, "weight_decay": weight_decay},
    ], head_params, backbone_params
