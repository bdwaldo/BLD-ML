from torchvision import transforms
import torch

def get_transforms(mean, std, rot_deg, noise_std, img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=ROT_DEG, translate=(0.05, 0.05),
                                 scale=(0.95, 1.05), fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t + torch.randn_like(t) * NOISE_STD),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=MEAN, std=STD),
        
    ])

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, val_transform
