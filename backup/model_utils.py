import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def create_model(device='cpu'):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    # Freeze backbone parameters
    for p in model.parameters():
        p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.requires_grad_(False)

    # Replace classifier head with single‑output neuron
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)

    return model.to(device)

def create_loss_and_optimizer(model, train_dataset, lr=1e-3, weight_decay=1e-4, device='cpu'):
    # Class imbalance handling
    cls_counts = Counter(train_dataset.targets)
    pos_weight = torch.tensor([cls_counts[0] / cls_counts[1]], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay
    )
    return criterion, optimizer
