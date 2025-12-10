#!/usr/bin/env python3
"""
Binary classification training script using EfficientNet_V2_S
Reflects the constants, transforms, dataset loading, and model setup
from the first code chunk you provided.
"""

import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ===================================================
# 1️⃣ Config
# ===================================================
BATCH_SIZE = 16
ROT_DEG = 90
NOISE_STD = 0.05
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()
IMG_SIZE = 384

# ImageNet normalization (EfficientNetV2 default)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================
# 2️⃣ Transforms
# ===================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=ROT_DEG, translate=(0.05, 0.05),
        scale=(0.95, 1.05), fill=0
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t + torch.randn_like(t) * NOISE_STD),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ===================================================
# 3️⃣ Dataset & DataLoaders
# ===================================================
data_root = '/90daydata/nematode_ml/BLD/NematodeDataset'

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_root, 'train'),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(data_root, 'val'),
    transform=val_transform
)

# Safety check: class order matches
assert set(train_dataset.classes) == set(val_dataset.classes), \
    "Class names differ between train and val."

# Remap val set to train set's ordering
val_to_train = {
    val_dataset.class_to_idx[label]: train_dataset.class_to_idx[label]
    for label in train_dataset.classes
}
val_dataset.samples = [(p, val_to_train[y]) for (p, y) in val_dataset.samples]
val_dataset.targets = [y for _, y in val_dataset.samples]
val_dataset.classes = train_dataset.classes
val_dataset.class_to_idx = train_dataset.class_to_idx

CLASSES = train_dataset.classes
class_mapping = train_dataset.class_to_idx
print(f"Classes: {CLASSES}")
print(f"Mapping: {class_mapping}")

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=True
)
valloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=True
)

# ===================================================
# 4️⃣ Model setup
# ===================================================
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)

# Freeze backbone parameters & batchnorm stats
for p in model.parameters():
    p.requires_grad = False
for m in model.features.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.requires_grad_(False)

# Replace classifier head with single output neuron
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 1)
model = model.to(DEVICE)

# Class imbalance handling
cls_counts = Counter(train_dataset.targets)
pos_weight = torch.tensor(
    [cls_counts[0] / cls_counts[1]],
    device=DEVICE
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer — only over trainable params
optimizer = optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-3, weight_decay=1e-4
)

print(f"Model ready on {DEVICE}, parameters to train: "
      f"{sum(p.requires_grad for p in model.parameters())}")

# ===================================================
# 5️⃣ Metrics, Grad-CAM helpers, and training loop
# ===================================================
import time
import copy
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

try:
    import cv2
except Exception:
    cv2 = None


def denorm_img(t: torch.Tensor, mean, std):
    """
    t: CHW float tensor normalized by mean/std in [0,1] space.
    returns HWC uint8 image in [0,255]
    """
    t = t.detach().cpu().float()
    for c in range(t.size(0)):
        t[c] = t[c] * std[c] + mean[c]
    t = t.clamp(0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def overlay_cam_on_image(img_hwc_uint8, cam_01, alpha=0.35):
    """
    img_hwc_uint8: HWC uint8 (0..255)
    cam_01: HxW float in [0,1]
    returns HWC uint8 overlay or None if cv2 unavailable
    """
    if cv2 is None:
        return None
    cam_255 = (cam_01 * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(cam_255, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat + (1 - alpha) * img_hwc_uint8).clip(0, 255).astype(np.uint8)
    return overlay


class GradCAM:
    """
    Minimal Grad-CAM for CNNs.
    target_layer: a module whose activations will be used (e.g., model.features[-1])
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.h_act = None
        self.h_grad = None
        self.acts = None
        self.grads = None
        self._register()

    def _register(self):
        def fwd_hook(module, inp, out):
            self.acts = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.grads = grad_out[0].detach()
        self.h_act = self.target_layer.register_forward_hook(fwd_hook)
        self.h_grad = self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, model, x, target_spec=None):
        """
        x: [1,C,H,W]
        target_spec:
          - None: uses the logit for positive class (index 0 for shape [N,1])
          - int: class index to target (for multi-class); for binary, use 0
        returns: [H,W] float in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # [1,1] for our head
        if logits.ndim == 2 and logits.size(1) == 1:
            target = logits[:, 0]
        else:
            idx = 0 if target_spec is None else int(target_spec)
            target = logits[:, idx]
        target.backward(retain_graph=True)

        A = self.acts  # [1, C, Hc, Wc]
        G = self.grads  # [1, C, Hc, Wc]
        if A is None or G is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = G.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=True)  # [1,1,Hc,Wc]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam

    def close(self):
        try:
            if self.h_act is not None:
                self.h_act.remove()
            if self.h_grad is not None:
                self.h_grad.remove()
        except Exception:
            pass


def train_1_binary(
    model,
    trainloader,
    valloader,
    criterion,
    optimizer,
    device=DEVICE,
    num_epochs=20,
    print_lr=True,
    early_stopping_patience=8,
    enable_gradcam=False,
    gradcam_every_n_epochs=None,  # if None, only when metric improves
    gradcam_samples=8,
    gradcam_outdir="./gradcam",
    gradcam_alpha=0.35,
    gradcam_input_mean=MEAN,
    gradcam_input_std=STD,
    gradcam_target_layer=None,    # e.g., model.features[-1]
    gradcam_target=None,          # None -> positive logit for our binary head
    scheduler=None,               # ReduceLROnPlateau on AUC if None
):
    os.makedirs(gradcam_outdir, exist_ok=True)

    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=False
        )

    acc_hist_train, acc_hist_val = [], []
    loss_hist_train, loss_hist_val = [], []

    best_metric_value = None
    best_state = None
    best_epoch = -1
    best_precision = best_recall = best_f1 = best_auc = float("nan")
    best_fpr, best_tpr = np.array([]), np.array([])

    auc_flip_warned = False
    epochs_since_improve = 0

    for epoch in range(num_epochs):
        t0 = time.time()
        # -------------------------------
        # Train
        # -------------------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)  # [B,1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().view(-1)
                correct += (preds == labels.view(-1).long()).sum().item()
                total += labels.size(0)
                running_loss += loss.item() * labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        loss_hist_train.append(epoch_loss)
        acc_hist_train.append(epoch_acc)

        # -------------------------------
        # Validate (gather probs and labels)
        # -------------------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs_pos = []
        val_cache_imgs = []
        val_cache_labels = []

        with torch.no_grad():
            for images, labels in valloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)  # positive class score as-is
                preds = (probs >= 0.5).long().view(-1)

                val_running_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels.view(-1).long()).sum().item()
                val_total += labels.size(0)

                all_labels.extend(labels.view(-1).long().cpu().tolist())
                all_probs_pos.extend(probs.view(-1).cpu().tolist())

                # cache some samples for Grad-CAM
                if enable_gradcam and len(val_cache_imgs) < gradcam_samples:
                    take = min(gradcam_samples - len(val_cache_imgs), images.size(0))
                    val_cache_imgs.append(images[:take].cpu())
                    val_cache_labels.extend(labels[:take].view(-1).long().cpu().tolist())

        val_loss = val_running_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        loss_hist_val.append(val_loss)
        acc_hist_val.append(val_acc)

        # -------------------------------
        # AUC flip safety check
        # -------------------------------
        try:
            auc_as_is = roc_auc_score(all_labels, all_probs_pos)
            auc_flipped = roc_auc_score(all_labels, 1 - np.array(all_probs_pos))
            use_flipped = auc_flipped > auc_as_is
            scores_for_auc = (1 - np.array(all_probs_pos)) if use_flipped else np.array(all_probs_pos)
            auc = max(auc_as_is, auc_flipped)
            if use_flipped and not auc_flip_warned:
                print("[train_1_binary] WARNING: AUC flip used — check pos_label or head output mapping")
                auc_flip_warned = True
        except ValueError:
            use_flipped = False
            scores_for_auc = np.array([])
            auc = float("nan")

        # Threshold tuning for best F1 (use same score direction as AUC)
        best_thr = 0.5
        if len(set(all_labels)) == 2 and scores_for_auc.size > 0:
            thresholds = np.linspace(0.0, 1.0, 101)
            f1s = []
            labels_np = np.array(all_labels)
            for t in thresholds:
                preds_t = (scores_for_auc >= t).astype(int)
                f1s.append(f1_score(labels_np, preds_t, zero_division=0))
            best_thr = float(thresholds[int(np.argmax(f1s))])

        preds_thr = (scores_for_auc >= best_thr).astype(int) if scores_for_auc.size > 0 else np.zeros_like(all_labels)
        precision = precision_score(all_labels, preds_thr, zero_division=0)
        recall = recall_score(all_labels, preds_thr, zero_division=0)
        f1 = f1_score(all_labels, preds_thr, zero_division=0)

        if scores_for_auc.size > 0 and len(set(all_labels)) == 2:
            fpr, tpr, _ = roc_curve(all_labels, scores_for_auc)
        else:
            fpr, tpr = np.array([]), np.array([])

        # -------------------------------
        # Metric tracking / checkpoint
        # -------------------------------
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

        scheduler.step(sched_value)

        lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups]) if print_lr else "hidden"
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"acc: {acc_hist_train[epoch]:.4f} | val_acc: {acc_hist_val[epoch]:.4f} | "
            f"loss: {loss_hist_train[epoch]:.4f} | val_loss: {loss_hist_val[epoch]:.4f} | "
            f"prec: {precision:.4f} | rec: {recall:.4f} | "
            f"f1: {f1:.4f} | auc: {auc:.4f} | thr: {best_thr:.2f} | lr: [{lrs}] | "
            f"epoch_time: {time.time() - t0:.1f}s"
        )

        # -------------------------------
        # Grad-CAM visualization trigger
        # -------------------------------
        trigger = False
        if enable_gradcam:
            if gradcam_every_n_epochs is not None:
                trigger = ((epoch + 1) % max(1, gradcam_every_n_epochs) == 0)
            else:
                trigger = is_better  # only when improved

        grabbed = sum(t.size(0) for t in val_cache_imgs) if enable_gradcam else 0
        if enable_gradcam and trigger and grabbed > 0:
            cache_imgs = torch.cat(val_cache_imgs, dim=0)  # [K, C, H, W]
            model.eval()
            target_layer = gradcam_target_layer if gradcam_target_layer is not None else model.features[-1]
            gc = GradCAM(model, target_layer)

            for i in range(min(gradcam_samples, cache_imgs.size(0))):
                x = cache_imgs[i:i+1].to(device)

                cam = gc.generate(model, x, target_spec=gradcam_target)  # [H,W] in [0,1]
                cam_np = cam.detach().cpu().numpy()

                img_np = denorm_img(cache_imgs[i], gradcam_input_mean, gradcam_input_std)

                overlay = overlay_cam_on_image(img_np, cam_np, alpha=gradcam_alpha)
                lab = val_cache_labels[i] if i < len(val_cache_labels) else -1
                out_name = f"epoch{epoch+1:03d}_idx{i}_label{int(lab)}.png"
                out_path = os.path.join(gradcam_outdir, out_name)

                from imageio import imwrite
                if overlay is None:
                    # fallback: save heatmap only if OpenCV not available
                    heat_gray = (cam_np * 255).astype(np.uint8)
                    if cv2 is not None:
                        heat_gray = cv2.resize(heat_gray, (img_np.shape[1], img_np.shape[0]))
                    imwrite(out_path, heat_gray)
                else:
                    imwrite(out_path, overlay)
            gc.close()
            print(f"[train_1_binary] Saved Grad-CAM overlays to {gradcam_outdir}")

        if early_stopping_patience and epochs_since_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

    # ---- Restore best model state ----
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
