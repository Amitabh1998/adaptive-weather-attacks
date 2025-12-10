"""
Data loading module for GTSRB dataset.
"""

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from .transforms import get_train_transforms, get_test_transforms

# Default configuration
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224


def get_dataloaders(data_dir: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """
    Create train, validation, and test dataloaders for GTSRB dataset.
    
    Args:
        data_dir: Path to GTSRB dataset directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=test_transform
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=test_transform
    )
    
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
    
    return train_loader, val_loader, test_loader


class RawImageDataset(Dataset):
    """
    Dataset that returns raw PIL images (for diffusion pipeline).
    """
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Path to image folder (e.g., data_dir/test)
        """
        self.dataset = datasets.ImageFolder(root_dir)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (PIL Image, label)
        """
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert('RGB')
        # Resize to standard size for diffusion
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return image, label


def get_raw_test_dataset(data_dir: str) -> RawImageDataset:
    """
    Get test dataset that returns raw PIL images (for diffusion pipeline).
    
    Args:
        data_dir: Path to GTSRB dataset directory
        
    Returns:
        RawImageDataset instance
    """
    return RawImageDataset(os.path.join(data_dir, 'test'))


__all__ = [
    'get_dataloaders',
    'get_raw_test_dataset',
    'RawImageDataset',
    'BATCH_SIZE',
    'NUM_WORKERS',
    'IMAGE_SIZE',
]
