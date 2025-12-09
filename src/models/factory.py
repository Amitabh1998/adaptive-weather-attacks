"""
Model factory for creating different classifier architectures.

Supports:
- ResNet-50 (torchvision)
- EfficientNet-B0 (timm)
- Vision Transformer (ViT) (transformers)
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from transformers import ViTForImageClassification

from ..config import DEVICE, NUM_CLASSES, CHECKPOINT_DIR


AVAILABLE_MODELS = ["resnet50", "efficientnet_b0", "vit"]


def get_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    device: str = DEVICE,
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_name: One of 'resnet50', 'efficientnet_b0', 'vit'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        
    Returns:
        Model loaded on specified device
        
    Example:
        >>> model = get_model('resnet50', num_classes=43)
        >>> output = model(torch.randn(1, 3, 224, 224).to('cuda'))
    """
    model_name = model_name.lower()
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {AVAILABLE_MODELS}"
        )
    
    if model_name == "resnet50":
        model = _create_resnet50(num_classes, pretrained)
    
    elif model_name == "efficientnet_b0":
        model = _create_efficientnet(num_classes, pretrained)
    
    elif model_name == "vit":
        model = _create_vit(num_classes, pretrained)
    
    return model.to(device)


def _create_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    """Create ResNet-50 model."""
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def _create_efficientnet(num_classes: int, pretrained: bool) -> nn.Module:
    """Create EfficientNet-B0 model."""
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def _create_vit(num_classes: int, pretrained: bool) -> nn.Module:
    """Create Vision Transformer model."""
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # Reset weights if not using pretrained
        model.apply(_reset_weights)
    
    return model


def _reset_weights(m: nn.Module) -> None:
    """Reset module weights."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    device: str = DEVICE,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Load a model from checkpoint.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of classes
        device: Device to load on
        checkpoint_path: Path to checkpoint (defaults to standard location)
        
    Returns:
        Loaded model
    """
    # Create model architecture
    model = get_model(model_name, num_classes=num_classes, pretrained=False, device="cpu")
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    return model.to(device)


def load_all_models(
    model_names: list = AVAILABLE_MODELS,
    device: str = DEVICE,
) -> dict:
    """
    Load all trained models.
    
    Args:
        model_names: List of model names to load
        device: Device to load on
        
    Returns:
        Dictionary mapping model names to loaded models
    """
    from .wrappers import ModelWrapper
    
    models_dict = {}
    
    for name in model_names:
        try:
            model = load_checkpoint(name, device=device)
            model = ModelWrapper(model)  # Wrap for consistent interface
            model.eval()
            models_dict[name] = model
            print(f"✓ Loaded {name}")
        except FileNotFoundError:
            print(f"✗ Checkpoint not found for {name}")
    
    return models_dict
