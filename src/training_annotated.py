#training.py

import time
from typing import Optional
import torch
import torch.nn as nn
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from optim_utils import (
    set_backbone_bn_eval,
    freeze_backbone,
    unfreeze_last_blocks,
    get_param_groups
)


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
    num_unfreeze_blocks: int = 2,            # renamed to avoid shadowing helper
    backbone_lr_factor: float = 0.1,
    freeze_backbone_bn: bool = True,
    print_lr: bool = True,
    pos_weight: Optional[float] = None,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gradcam_target_layer: Optional[str] = None,
    save_path: Optional[str] = None,
    metrics_save_path: Optional[str] = None,
):
    model.to(device)

    # Freeze backbone initially
    freeze_backbone(model, freeze_bn=freeze_backbone_bn)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).to(device) if pos_weight else None
        )

    if optimizer is None:
        param_groups, _, _ = get_param_groups(model, backbone_lr_factor, base_lr, weight_decay)
        optimizer = torch.optim.AdamW(param_groups)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    scaler = GradScaler(enabled=mixed_precision)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(num_epochs):
        # Scheduled unfreeze
        if unfreeze_at_epoch is not None and epoch == unfreeze_at_epoch:
            print(f"🔓 Unfreezing last {num_unfreeze_blocks} backbone blocks at epoch {epoch}")
            unfreeze_last_blocks(model, n_blocks=num_unfreeze_blocks, freeze_bn=freeze_backbone_bn)

            # Rebuild optimizer and scheduler to include new params
            param_groups, _, _ = get_param_groups(model, backbone_lr_factor, base_lr, weight_decay)
            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        start_time = time.time()
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast(enabled=mixed_precision):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # keep only main logits
                loss = criterion(outputs.squeeze(), targets.float())



            scaler.scale(loss).backward()

            if grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds.squeeze() == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true_all, y_pred_all, y_score_all = [], [], []

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(enabled=mixed_precision):
                    outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # keep only main logits
                loss = criterion(outputs.squeeze(), targets.float())


                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                y_true_all.extend(targets.cpu().numpy())
                y_pred_all.extend(preds)
                y_score_all.extend(probs)

                val_correct += (preds.squeeze() == targets.cpu().numpy()).sum()
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_prec = precision_score(y_true_all, y_pred_all, zero_division=0)
        val_rec = recall_score(y_true_all, y_pred_all, zero_division=0)
        val_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        try:
            val_auroc = roc_auc_score(y_true_all, y_score_all)
        except ValueError:
            val_auroc = float('nan')

        scheduler.step(val_loss)

        if print_lr:
            print(f"Epoch {epoch+1}/{num_epochs} - LRs: {[pg['lr'] for pg in optimizer.param_groups]}")

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}] "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} | "
              f"P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}, AUROC: {val_auroc:.4f} "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_prec": val_prec,
            "val_rec": val_rec,
            "val_f1": val_f1,
            "val_auroc": val_auroc
        })

        if metrics_save_path:
            pd.DataFrame(history).to_csv(metrics_save_path, index=False)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "final_model.pth")
    print("✅ Final model weights saved to final_model.pth")
    return history
