# src/training.py
from __future__ import annotations
import copy
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from metrics import auc_with_flip, tune_threshold
from model_utils import freeze_backbone, set_backbone_bn_eval, unfreeze_last_blocks
from optim_utils import build_optimizer, build_plateau_scheduler


def train_1_binary(
    model: nn.Module,
    num_epochs: int,
    trainloader,
    valloader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    mixed_precision: bool = False,
    early_stopping_patience: Optional[int] = None,
    pos_label: Optional[int] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip_norm: Optional[float] = 1.0,
    unfreeze_at_epoch: Optional[int] = None,
    unfreeze_last_blocks: int = 2,
    backbone_lr_factor: float = 0.1,
    freeze_backbone_bn: bool = True,
    print_lr: bool = True,
    pos_weight: Optional[float] = None,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    # Device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Criterion
    if criterion is None:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    # Positive label heuristic (for logging only)
    if pos_label is None:
        ds = getattr(trainloader, "dataset", None)
        if ds is not None and hasattr(ds, "class_to_idx"):
            if "positive" in ds.class_to_idx:
                pos_label = ds.class_to_idx["positive"]
            elif 1 in ds.class_to_idx.values():
                pos_label = 1
            else:
                pos_label = 1
        else:
            pos_label = 1
        print(f"[train_1_binary] Using pos_label={pos_label}")

    # AMP scaler
    scaler = GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    # Freeze backbone initially (head fine-tunes first)
    freeze_backbone(model, freeze_bn=freeze_backbone_bn)

    # Optimizer and scheduler
    optimizer = optimizer or build_optimizer(
        model, backbone_lr_factor=backbone_lr_factor, lr=base_lr, weight_decay=weight_decay
    )
    scheduler = scheduler or build_plateau_scheduler(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    # History and tracking
    loss_hist_train, acc_hist_train = [0.0] * num_epochs, [0.0] * num_epochs
    loss_hist_val, acc_hist_val = [0.0] * num_epochs, [0.0] * num_epochs

    best_state, best_epoch, best_metric_value = None, -1, None
    best_precision = best_recall = best_f1 = float("nan")
    best_auc = float("nan")
    best_fpr = np.array([])
    best_tpr = np.array([])
    auc_flip_warned = False
    epochs_since_improve = 0

    for epoch in range(num_epochs):
        # Optional backbone unfreeze at a given epoch (unfreezing last N blocks)
        if unfreeze_at_epoch is not None and epoch == unfreeze_at_epoch:
            unfreeze_last_blocks(model, n_blocks=unfreeze_last_blocks, freeze_bn=freeze_backbone_bn)
            print(f"[train_1_binary] Unfroze last {unfreeze_last_blocks} blocks at epoch {epoch + 1}")

        # Train
        model.train()
        if freeze_backbone_bn and hasattr(model, "features"):
            model.features.apply(set_backbone_bn_eval)

        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for xb, yb in trainloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()  # shape (N,)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
                with autocast(dtype=autocast_dtype):
                    logits = model(xb).squeeze(1)  # (N,)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == yb.long()).sum().item()
            train_loss_sum += loss.item() * yb.size(0)
            train_total += yb.size(0)

        loss_hist_train[epoch] = train_loss_sum / train_total
        acc_hist_train[epoch] = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        all_labels: List[float] = []
        all_probs_pos: List[float] = []

        with torch.inference_mode():
            for xb, yb in valloader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)

                probs_pos = torch.sigmoid(logits)
                preds = (probs_pos >= 0.5).long()

                val_correct += (preds == yb.long()).sum().item()
                val_loss_sum += loss.item() * yb.size(0)
                val_total += yb.size(0)

                all_labels.extend(yb.detach().cpu().numpy().tolist())
                all_probs_pos.extend(probs_pos.detach().cpu().numpy().tolist())

        loss_hist_val[epoch] = val_loss_sum / val_total
        acc_hist_val[epoch] = val_correct / val_total

        # AUC with flip safety
        auc, use_flipped, scores_for_auc, fpr, tpr = auc_with_flip(np.array(all_labels), np.array(all_probs_pos))
        if use_flipped and not auc_flip_warned:
            print("[train_1_binary] WARNING: AUC flip used — check pos_label or head output mapping")
            auc_flip_warned = True

        # Threshold tuning
        best_thr, precision, recall, f1 = tune_threshold(np.array(all_labels), np.array(all_probs_pos))

        # Improvement check (monitor AUC)
        current_metric, sched_value = auc, auc
        is_better = (best_metric_value is None) or \
                    (np.isnan(best_metric_value) and not np.isnan(current_metric)) or \
                    (current_metric > best_metric_value)

        if is_better:
            best_metric_value = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_precision, best_recall, best_f1, best_auc = precision, recall, f1, auc
            best_fpr, best_tpr = fpr, tpr
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # Step scheduler with monitored metric
        scheduler.step(sched_value)

        # Log line
        lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups]) if print_lr else "hidden"
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"acc: {acc_hist_train[epoch]:.4f} | val_acc: {acc_hist_val[epoch]:.4f} | "
            f"loss: {loss_hist_train[epoch]:.4f} | val_loss: {loss_hist_val[epoch]:.4f} | "
            f"prec: {precision:.4f} | rec: {recall:.4f} | "
            f"f1: {f1:.4f} | auc: {auc:.4f} | thr: {best_thr:.2f} | lr: [{lrs}]"
        )

        # Early stopping
        if early_stopping_patience and epochs_since_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"Best epoch: {best_epoch+1} | "
        f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, "
        f"F1: {best_f1:.4f}, ROC AUC: {best_auc:.4f}"
    )

    return (
        loss_hist_train, loss_hist_val,
        acc_hist_train, acc_hist_val,
        best_precision, best_recall, best_f1,
        best_fpr, best_tpr, best_auc
    )
