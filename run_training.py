# scripts/run_training.py
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from torchvision import transforms

from training import train_1_binary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(f"{args.data_root}/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0)

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)

    train_1_binary(
        model=model,
        num_epochs=args.epochs,
        trainloader=train_loader,
        valloader=val_loader,
        device=device,
        mixed_precision=args.mixed_precision,
        early_stopping_patience=5,
        backbone_lr_factor=0.1,
        freeze_backbone_bn=True,
        grad_clip_norm=1.0,
        print_lr=True
    )

if __name__ == "__main__":
    main()
