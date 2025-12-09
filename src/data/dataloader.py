"""
DataLoader factory functions for GTSRB dataset.
"""

from typing import Tuple, Optional

from torch.utils.data import DataLoader, random_split

from .gtsrb_dataset import GTSRBDataset
from .transforms import get_train_transforms, get_test_transforms
from ..config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    VAL_SPLIT,
    DATA_DIR,
)


def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to GTSRB dataset (defaults to config.DATA_DIR)
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders()
        >>> for images, labels in train_loader:
        ...     # Train step
    """
    if data_dir is None:
        data_dir = str(DATA_DIR)
    
    # Load full training set
    full_train_dataset = GTSRBDataset(
        root_dir=data_dir,
        split="train",
        transform=get_train_transforms()
    )
    
    # Split into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )
    
    # Load test set
    test_dataset = GTSRBDataset(
        root_dir=data_dir,
        split="test",
        transform=get_test_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  - Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  - Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  - Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def get_test_loader(
    data_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Create test dataloader only.
    
    Args:
        data_dir: Path to GTSRB dataset
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Test DataLoader
    """
    if data_dir is None:
        data_dir = str(DATA_DIR)
    
    test_dataset = GTSRBDataset(
        root_dir=data_dir,
        split="test",
        transform=get_test_transforms()
    )
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )


def get_raw_test_dataset(data_dir: Optional[str] = None) -> GTSRBDataset:
    """
    Get test dataset without any transforms (for diffusion input).
    
    Args:
        data_dir: Path to GTSRB dataset
        
    Returns:
        GTSRBDataset with no transforms
    """
    if data_dir is None:
        data_dir = str(DATA_DIR)
    
    return GTSRBDataset(
        root_dir=data_dir,
        split="test",
        transform=None  # Return PIL images
    )
