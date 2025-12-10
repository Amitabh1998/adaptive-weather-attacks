"""
Data loading module for GTSRB dataset (raw format).
"""

import os
import csv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import torch

from .transforms import get_train_transforms, get_test_transforms

# Default configuration
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224
VAL_SPLIT = 0.1  # 10% of training data for validation


class GTSRBTestDataset(Dataset):
    """
    GTSRB Test Dataset that reads labels from CSV file.
    """
    def __init__(self, images_dir: str, csv_path: str, transform=None):
        """
        Args:
            images_dir: Path to GTSRB_final_test_images folder
            csv_path: Path to GT-final_test.csv
            transform: Optional transform to apply
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Read CSV file to get image -> label mapping
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header
            for row in reader:
                filename = row[0]  # First column is filename
                label = int(row[7])  # 8th column is ClassId
                self.samples.append((filename, label))
        
        print(f"  Loaded {len(self.samples)} test samples from CSV")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class RawImageDataset(Dataset):
    """
    Dataset that returns raw PIL images (for diffusion pipeline).
    Works with GTSRB test format.
    """
    def __init__(self, images_dir: str, csv_path: str):
        """
        Args:
            images_dir: Path to GTSRB_final_test_images folder
            csv_path: Path to GT-final_test.csv
        """
        self.images_dir = images_dir
        
        # Read CSV file to get image -> label mapping
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header
            for row in reader:
                filename = row[0]
                label = int(row[7])
                self.samples.append((filename, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (PIL Image resized to 224x224, label)
        """
        filename, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert('RGB')
        # Resize to standard size for diffusion
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return image, label


def get_dataloaders(data_dir: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """
    Create train, validation, and test dataloaders for GTSRB dataset.
    
    Handles raw GTSRB format:
    - data_dir/GTSRB_final_training_images/ (class folders: 00000, 00001, ...)
    - data_dir/GTSRB_final_test_images/ (flat .ppm files)
    - data_dir/GTSRB_Final_Test_GT/GT-final_test.csv
    
    Args:
        data_dir: Path to GTSRB dataset directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Training data (already in class folders)
    train_dir = os.path.join(data_dir, 'GTSRB_final_training_images')
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Split training into train/val
    train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset_raw = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create val dataset with test transforms (no augmentation)
    val_dataset_folder = datasets.ImageFolder(train_dir, transform=test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset_folder, val_dataset_raw.indices)
    
    # Test data (flat folder with CSV labels)
    test_images_dir = os.path.join(data_dir, 'GTSRB_final_test_images')
    test_csv_path = os.path.join(data_dir, 'GTSRB_Final_Test_GT', 'GT-final_test.csv')
    test_dataset = GTSRBTestDataset(test_images_dir, test_csv_path, transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Classes: {len(full_train_dataset.classes)}")
    
    return train_loader, val_loader, test_loader


def get_raw_test_dataset(data_dir: str) -> RawImageDataset:
    """
    Get test dataset that returns raw PIL images (for diffusion pipeline).
    
    Args:
        data_dir: Path to GTSRB dataset directory
        
    Returns:
        RawImageDataset instance
    """
    test_images_dir = os.path.join(data_dir, 'GTSRB_final_test_images')
    test_csv_path = os.path.join(data_dir, 'GTSRB_Final_Test_GT', 'GT-final_test.csv')
    return RawImageDataset(test_images_dir, test_csv_path)


__all__ = [
    'get_dataloaders',
    'get_raw_test_dataset',
    'GTSRBTestDataset',
    'RawImageDataset',
    'BATCH_SIZE',
    'NUM_WORKERS',
    'IMAGE_SIZE',
]
