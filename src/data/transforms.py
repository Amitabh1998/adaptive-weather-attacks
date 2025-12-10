"""
Image transforms for GTSRB dataset.
"""

from torchvision import transforms

# ImageNet normalization (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms():
    """
    Get training transforms with data augmentation.
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms():
    """
    Get test/validation transforms (no augmentation).
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_unnormalize_transform():
    """
    Get transform to unnormalize images back to [0, 1] range.
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )


__all__ = [
    'get_train_transforms',
    'get_test_transforms',
    'get_unnormalize_transform',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'IMAGE_SIZE',
]
