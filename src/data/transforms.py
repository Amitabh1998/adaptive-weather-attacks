"""
Data transforms and augmentation for traffic sign images.
"""

from torchvision import transforms

from ..config import IMAGE_SIZE, MEAN, STD


def get_train_transforms() -> transforms.Compose:
    """
    Get training transforms with data augmentation.
    
    Returns:
        Composed transform for training images
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Get test/validation transforms (no augmentation).
    
    Returns:
        Composed transform for test/validation images
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_denormalize_transform() -> transforms.Compose:
    """
    Get transform to convert normalized tensor back to displayable image.
    
    Returns:
        Transform to denormalize images
    """
    inv_mean = [-m / s for m, s in zip(MEAN, STD)]
    inv_std = [1 / s for s in STD]
    
    return transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=inv_std),
        transforms.Normalize(mean=inv_mean, std=[1, 1, 1]),
    ])


def denormalize_image(tensor):
    """
    Denormalize a single image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        Denormalized numpy array [H, W, C] in range [0, 1]
    """
    import numpy as np
    import torch
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().detach().cpu()
    
    # Denormalize
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    
    # Clamp and convert
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy [H, W, C]
    return tensor.permute(1, 2, 0).numpy()


def tensor_to_pil(tensor):
    """
    Convert a normalized tensor to PIL Image.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        PIL Image
    """
    from PIL import Image
    import numpy as np
    
    img_np = denormalize_image(tensor)
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)
