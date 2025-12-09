"""
Pixel-based adversarial attacks using torchattacks library.

Implements FGSM, PGD, and C&W attacks for baseline comparison
against weather-based attacks.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchattacks

from ..config import (
    DEVICE,
    FGSM_EPS,
    PGD_EPS,
    PGD_ALPHA,
    PGD_STEPS,
    CW_C,
    CW_KAPPA,
    CW_STEPS,
    CW_LR,
)


def create_fgsm_attack(
    model: nn.Module,
    eps: float = FGSM_EPS,
) -> torchattacks.Attack:
    """
    Create FGSM (Fast Gradient Sign Method) attack.
    
    Args:
        model: Target model
        eps: Perturbation budget (L-infinity)
        
    Returns:
        FGSM attack instance
    """
    return torchattacks.FGSM(model, eps=eps)


def create_pgd_attack(
    model: nn.Module,
    eps: float = PGD_EPS,
    alpha: float = PGD_ALPHA,
    steps: int = PGD_STEPS,
) -> torchattacks.Attack:
    """
    Create PGD (Projected Gradient Descent) attack.
    
    Args:
        model: Target model
        eps: Perturbation budget (L-infinity)
        alpha: Step size
        steps: Number of iterations
        
    Returns:
        PGD attack instance
    """
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)


def create_cw_attack(
    model: nn.Module,
    c: float = CW_C,
    kappa: float = CW_KAPPA,
    steps: int = CW_STEPS,
    lr: float = CW_LR,
) -> torchattacks.Attack:
    """
    Create C&W (Carlini-Wagner) attack.
    
    Args:
        model: Target model
        c: Confidence parameter
        kappa: Minimum confidence
        steps: Number of optimization steps
        lr: Learning rate
        
    Returns:
        C&W attack instance
    """
    return torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)


def evaluate_attack(
    model: nn.Module,
    attack: torchattacks.Attack,
    loader: DataLoader,
    device: str = DEVICE,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Evaluate attack effectiveness.
    
    Args:
        model: Target model
        attack: Attack instance
        loader: Data loader
        device: Device to run on
        max_batches: Maximum batches to evaluate (None for all)
        
    Returns:
        Dictionary with attack metrics
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Evaluating attack", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            if hasattr(clean_outputs, 'logits'):
                clean_outputs = clean_outputs.logits
            _, clean_preds = torch.max(clean_outputs, 1)
            clean_correct += (clean_preds == labels).sum().item()
        
        # Generate adversarial examples
        adv_images = attack(images, labels)
        
        # Adversarial accuracy
        with torch.no_grad():
            adv_outputs = model(adv_images)
            if hasattr(adv_outputs, 'logits'):
                adv_outputs = adv_outputs.logits
            _, adv_preds = torch.max(adv_outputs, 1)
            adv_correct += (adv_preds == labels).sum().item()
        
        total += labels.size(0)
        
        pbar.set_postfix({
            'clean': f'{100 * clean_correct / total:.1f}%',
            'adv': f'{100 * adv_correct / total:.1f}%'
        })
    
    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total
    attack_success_rate = 100 - adv_acc
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': attack_success_rate,
        'total_samples': total,
    }


def run_attack_comparison(
    model: nn.Module,
    loader: DataLoader,
    attacks: Optional[Dict] = None,
    device: str = DEVICE,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Run comparison of multiple attacks.
    
    Args:
        model: Target model
        loader: Data loader
        attacks: Dictionary of attack name to attack instance
                 (defaults to FGSM, PGD, CW)
        device: Device to run on
        max_batches: Maximum batches per attack
        
    Returns:
        Dictionary with results for each attack
    """
    if attacks is None:
        attacks = {
            'FGSM': create_fgsm_attack(model),
            'PGD': create_pgd_attack(model),
            'CW': create_cw_attack(model),
        }
    
    results = {}
    
    print("=" * 60)
    print("ATTACK COMPARISON")
    print("=" * 60)
    
    for attack_name, attack in attacks.items():
        print(f"\n{attack_name}:")
        metrics = evaluate_attack(model, attack, loader, device, max_batches)
        results[attack_name] = metrics
        
        print(f"  Clean Accuracy: {metrics['clean_accuracy']:.2f}%")
        print(f"  Adversarial Accuracy: {metrics['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    
    return results


def per_class_accuracy(
    model: nn.Module,
    attack: torchattacks.Attack,
    loader: DataLoader,
    num_classes: int = 43,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Calculate per-class adversarial accuracy.
    
    Args:
        model: Target model
        attack: Attack instance
        loader: Data loader
        num_classes: Number of classes
        device: Device to run on
        
    Returns:
        Array of per-class accuracies
    """
    model.eval()
    
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    
    for images, labels in tqdm(loader, desc="Per-class eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
        
        for gt, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            total[gt] += 1
            if pred == gt:
                correct[gt] += 1
    
    # Avoid division by zero
    per_class_acc = 100 * correct / np.maximum(total, 1)
    
    return per_class_acc
