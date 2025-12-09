"""
GTSRB Dataset class for loading German Traffic Sign Recognition Benchmark.

Handles the official GTSRB folder structure:
- Training: GTSRB_final_training_images/XXXXX/*.ppm
- Test: GTSRB_final_test_images/*.ppm + GT-final_test.csv
"""

import os
import glob
from typing import Optional, Callable, Tuple, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..config import NUM_CLASSES


class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for the German Traffic Sign Recognition Benchmark.
    
    Args:
        root_dir: Path to the GTSRB dataset directory
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        
    Example:
        >>> dataset = GTSRBDataset('/content/GTSRB_dataset', split='train')
        >>> image, label = dataset[0]
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images: List[str] = []
        self.labels: List[int] = []
        
        if split == "train":
            self._load_train_data()
        else:
            self._load_test_data()
    
    def _load_train_data(self) -> None:
        """Load training data from class-organized folders."""
        train_dir = os.path.join(self.root_dir, "GTSRB_final_training_images")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        for class_id in range(NUM_CLASSES):
            class_folder = os.path.join(train_dir, f"{class_id:05d}")
            
            if not os.path.exists(class_folder):
                continue
            
            image_files = glob.glob(os.path.join(class_folder, "*.ppm"))
            
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(class_id)
        
        print(f"✓ Loaded {len(self.images)} training images from {NUM_CLASSES} classes")
    
    def _load_test_data(self) -> None:
        """Load test data using GT-final_test.csv for labels."""
        test_dir = os.path.join(self.root_dir, "GTSRB_final_test_images")
        csv_path = os.path.join(self.root_dir, "GTSRB_Final_Test_GT", "GT-final_test.csv")
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Test CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path, sep=";")
        
        for _, row in df.iterrows():
            img_name = row["Filename"]
            class_id = int(row["ClassId"])
            img_path = os.path.join(test_dir, img_name)
            
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(class_id)
        
        print(f"✓ Loaded {len(self.images)} test images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the number of samples per class."""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def get_sample_path(self, idx: int) -> str:
        """Get the file path for a sample."""
        return self.images[idx]
