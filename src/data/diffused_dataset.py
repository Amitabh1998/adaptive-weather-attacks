"""
Dataset class for diffusion-generated adversarial images.
"""

from typing import Optional, Callable, Tuple, List
from PIL import Image
from torch.utils.data import Dataset


class DiffusedDataset(Dataset):
    """
    Dataset for pairs of original and diffusion-generated images.
    
    Used for evaluating weather attack effectiveness.
    
    Args:
        metadata: List of tuples (original_path, diffused_path, label, prompt)
        transform: Optional transform to apply to both images
        
    Example:
        >>> meta = [(orig_path, diff_path, label, "in heavy fog"), ...]
        >>> dataset = DiffusedDataset(meta, transform=get_test_transforms())
        >>> orig_img, diff_img, label, prompt = dataset[0]
    """
    
    def __init__(
        self,
        metadata: List[Tuple[str, str, int, str]],
        transform: Optional[Callable] = None
    ):
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple:
        orig_path, diff_path, label, prompt = self.metadata[idx]
        
        orig_img = Image.open(orig_path).convert("RGB")
        diff_img = Image.open(diff_path).convert("RGB")
        
        if self.transform is not None:
            orig_img = self.transform(orig_img)
            diff_img = self.transform(diff_img)
        
        return orig_img, diff_img, label, prompt


class AdversarialDataset(Dataset):
    """
    Dataset containing adversarial examples with metadata.
    
    Args:
        images: List of image tensors or paths
        labels: True labels
        predictions: Model predictions on adversarial images
        metadata: Optional additional metadata
        transform: Optional transform
    """
    
    def __init__(
        self,
        images: List,
        labels: List[int],
        predictions: Optional[List[int]] = None,
        metadata: Optional[List[dict]] = None,
        transform: Optional[Callable] = None
    ):
        self.images = images
        self.labels = labels
        self.predictions = predictions or [None] * len(images)
        self.metadata = metadata or [{}] * len(images)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        image = self.images[idx]
        
        # Handle both tensor and path inputs
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        
        return {
            "image": image,
            "label": self.labels[idx],
            "prediction": self.predictions[idx],
            "metadata": self.metadata[idx],
        }
    
    def get_attack_success_rate(self) -> float:
        """Calculate attack success rate (misclassification rate)."""
        if self.predictions[0] is None:
            raise ValueError("Predictions not available")
        
        misclassified = sum(
            1 for l, p in zip(self.labels, self.predictions) if l != p
        )
        return misclassified / len(self.labels)
