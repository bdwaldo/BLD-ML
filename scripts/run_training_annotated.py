#run_training.py
import os
import csv
import argparse #argparse: handles command-line arguments (like --epochs, --model).
import torch
import torch.nn as nn
import torchvision #torch / torchvision: deep learning framework and pretrained models.
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score #sklearn.metrics: calculates precision, recall, F1, AUROC.

from training import train_1_binary #train_1_binary: the custom training loop (defined in training.py).

from torch.utils.data import Dataset

class FlipLabels(Dataset):
    """
    Wrap a dataset and flip binary labels 0 <-> 1.
    Ensures the intended 'positive' class is encoded as 1 
    for training, loss, and metrics.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return img, 1 - label  # swap encoding


def build_model(model_name: str, num_classes: int = 1):
    """Factory to build one of several pretrained backbones for binary classification (efficientnet, resnet50, mobilenet, inception)."""
    import torch.nn as nn
    model_name = model_name.lower()

    if model_name == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes) #Replaces the final classification layer with a binary output (num_classes=1).
        gradcam_target_layer = "features[-1]" #Returns both the model and the layer to target for Grad-CAM visualization.

    elif model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        gradcam_target_layer = "layer4"

    elif model_name == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        gradcam_target_layer = "features[-1]"

    elif model_name == "inception_v3":
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        # Newer torchvision enforces aux_logits=True with pretrained weights
        model = inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        gradcam_target_layer = "Mixed_7c"

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, gradcam_target_layer


def main(): #Matches the flags you pass in your SLURM script (--data-root, --epochs, --model mobilenet_v3_large, etc.).
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--model", type=str, default="efficientnet_v2_s",
                    help="Backbone model: efficientnet_v2_s, resnet50, mobilenet_v3_large, inception_v3")
    ap.add_argument("--unfreeze-at-epoch", type=int, default=3)
    ap.add_argument("--num-unfreeze-blocks", type=int, default=2)
    args = ap.parse_args()

    # Adjust default img size for Inception if not overridden
    if args.model.lower() == "inception_v3" and args.img_size == 384:
        args.img_size = 299

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    #Training transformations
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    #For validation/test, only resize + normalize.
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    #Dataset setup
    train_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/val", transform=val_tf)
    test_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/test", transform=val_tf)

    # Create datasets with transforms
    train_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder(f"{args.data_root}/val", transform=val_tf)
    test_ds  = torchvision.datasets.ImageFolder(f"{args.data_root}/test", transform=val_tf)

    # Wrap datasets so 'BLD' becomes label 1 (positive)
    train_ds = FlipLabels(train_ds)
    val_ds   = FlipLabels(val_ds)
    test_ds  = FlipLabels(test_ds)

    #Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers, #Uses multiple workers (--num-workers) for parallel data loading.
        pin_memory=(device.type == "cuda"), #pin_memory=True if using GPU for faster transfer.
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0
    )

    # Build model dynamically
    model, gradcam_target_layer = build_model(args.model, num_classes=1)

    # Timestamped metrics file in outputs/
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_filename = f"outputs/metrics_{timestamp}.csv"

    history = train_1_binary( #Calls custom train_1_binary function to handle training loop, validation, checkpointing, and metrics logging.
        model=model,
        num_epochs=args.epochs,
        trainloader=train_loader,
        valloader=val_loader,
        device=device,
        mixed_precision=args.mixed_precision,
        early_stopping_patience=10,
        backbone_lr_factor=0.1,
        freeze_backbone_bn=True,
        grad_clip_norm=1.0,
        print_lr=True,
        gradcam_target_layer=gradcam_target_layer,
        save_path="outputs/best_model.pth", #Saves the best model to outputs/best_model.pth.
        metrics_save_path=metrics_filename,
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        num_unfreeze_blocks=args.num_unfreeze_blocks,
        base_lr=1e-3,
        weight_decay=1e-4
)
    


    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device)) #Reloads the best model.
    model.to(device)
    model.eval() #Evaluates on the test set.

    y_true, y_pred, y_score = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs.squeeze(), targets.float()) #Compute loss
            test_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds)
            y_score.extend(probs)

    test_loss /= len(test_loader.dataset)
    test_acc = (np.array(y_pred).squeeze() == np.array(y_true)).mean() #Compute accuracy
    test_prec = precision_score(y_true, y_pred, zero_division=0) #Compute precision
    test_rec = recall_score(y_true, y_pred, zero_division=0) #Compute recall
    test_f1 = f1_score(y_true, y_pred, zero_division=0) #Compute F1
    try:
        test_auroc = roc_auc_score(y_true, y_score) #Compute AUROC
    except ValueError:
        test_auroc = float('nan')

    print(f"\n📊 Test results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
          f"P: {test_prec:.4f}, R: {test_rec:.4f}, "
          f"F1: {test_f1:.4f}, AUROC: {test_auroc:.4f}")

    # Save test metrics to CSV
    test_metrics_path = f"outputs/test_metrics_{timestamp}.csv" #Save in outputs/test_metrics_<timestamp>.csv
    with open(test_metrics_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "acc", "precision", "recall", "f1", "auroc"])
        writer.writerow([test_loss, test_acc, test_prec, test_rec, test_f1, test_auroc])

    print(f"💾 Test metrics saved to {test_metrics_path}")


if __name__ == "__main__":
    main()
