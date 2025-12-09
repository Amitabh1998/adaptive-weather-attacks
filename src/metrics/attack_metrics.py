"""
Metrics for evaluating adversarial attack effectiveness.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..config import DEVICE


def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = DEVICE,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model: Classification model
        loader: Data loader
        device: Device to run on
        
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def compute_attack_success_rate(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Compute attack success rate and related metrics.
    
    Args:
        model: Target model
        clean_images: Original images
        adv_images: Adversarial images
        labels: True labels
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    clean_images = clean_images.to(device)
    adv_images = adv_images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        # Clean predictions
        clean_outputs = model(clean_images)
        if hasattr(clean_outputs, 'logits'):
            clean_outputs = clean_outputs.logits
        _, clean_preds = torch.max(clean_outputs, 1)
        
        # Adversarial predictions
        adv_outputs = model(adv_images)
        if hasattr(adv_outputs, 'logits'):
            adv_outputs = adv_outputs.logits
        _, adv_preds = torch.max(adv_outputs, 1)
    
    total = labels.size(0)
    clean_correct = (clean_preds == labels).sum().item()
    adv_correct = (adv_preds == labels).sum().item()
    
    # Attack success: originally correct, now incorrect
    originally_correct = (clean_preds == labels)
    now_incorrect = (adv_preds != labels)
    successful_attacks = (originally_correct & now_incorrect).sum().item()
    
    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total
    
    # ASR relative to correctly classified samples
    if clean_correct > 0:
        asr = 100 * successful_attacks / clean_correct
    else:
        asr = 0.0
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': asr,
        'total_samples': total,
        'successful_attacks': successful_attacks,
    }


def evaluate_model_on_images(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str = DEVICE,
) -> Tuple[float, torch.Tensor]:
    """
    Evaluate model on a batch of images.
    
    Args:
        model: Classification model
        images: Image tensor [B, C, H, W]
        labels: True labels [B]
        device: Device to run on
        
    Returns:
        Tuple of (accuracy, predictions)
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        _, predictions = torch.max(outputs, 1)
    
    correct = (predictions == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)
    
    return accuracy, predictions


def compute_transferability(
    source_model: nn.Module,
    target_models: Dict[str, nn.Module],
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: str = DEVICE,
) -> Dict[str, Dict]:
    """
    Compute attack transferability across models.
    
    Args:
        source_model: Model used to generate attacks
        target_models: Dictionary of models to test transfer
        clean_images: Original images
        adv_images: Adversarial images
        labels: True labels
        device: Device to run on
        
    Returns:
        Dictionary with transfer results for each model
    """
    results = {}
    
    # Source model results
    source_metrics = compute_attack_success_rate(
        source_model, clean_images, adv_images, labels, device
    )
    results['source'] = source_metrics
    
    # Target model results
    for name, model in target_models.items():
        metrics = compute_attack_success_rate(
            model, clean_images, adv_images, labels, device
        )
        results[name] = metrics
    
    return results


def per_class_attack_success(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 43,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Compute per-class attack success rate.
    
    Args:
        model: Target model
        clean_images: Original images
        adv_images: Adversarial images
        labels: True labels
        num_classes: Number of classes
        device: Device to run on
        
    Returns:
        Array of per-class attack success rates
    """
    model.eval()
    
    clean_images = clean_images.to(device)
    adv_images = adv_images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        clean_outputs = model(clean_images)
        adv_outputs = model(adv_images)
        
        if hasattr(clean_outputs, 'logits'):
            clean_outputs = clean_outputs.logits
        if hasattr(adv_outputs, 'logits'):
            adv_outputs = adv_outputs.logits
        
        _, clean_preds = torch.max(clean_outputs, 1)
        _, adv_preds = torch.max(adv_outputs, 1)
    
    # Per-class statistics
    class_total = np.zeros(num_classes)
    class_success = np.zeros(num_classes)
    
    labels_np = labels.cpu().numpy()
    clean_preds_np = clean_preds.cpu().numpy()
    adv_preds_np = adv_preds.cpu().numpy()
    
    for label, clean_pred, adv_pred in zip(labels_np, clean_preds_np, adv_preds_np):
        if clean_pred == label:  # Originally correct
            class_total[label] += 1
            if adv_pred != label:  # Now incorrect
                class_success[label] += 1
    
    # Avoid division by zero
    per_class_asr = 100 * class_success / np.maximum(class_total, 1)
    
    return per_class_asr
