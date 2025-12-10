"""
Pixel-based adversarial attacks using torchattacks library.

Implements FGSM, PGD, and C&W attacks for baseline comparison
against weather-based attacks.
"""

import torch
import torch.nn as nn
import torchattacks

# Default attack parameters
FGSM_EPS = 0.03
PGD_EPS = 0.03
PGD_ALPHA = 0.01
PGD_STEPS = 10
CW_C = 1
CW_KAPPA = 0
CW_STEPS = 50
CW_LR = 0.01


def create_fgsm_attack(
    model: nn.Module,
    eps: float = FGSM_EPS,
):
    """
    Create FGSM (Fast Gradient Sign Method) attack.
    
    Args:
        model: Target model to attack
        eps: Maximum perturbation (L-inf norm)
        
    Returns:
        torchattacks FGSM attack instance
    """
    return torchattacks.FGSM(model, eps=eps)


def create_pgd_attack(
    model: nn.Module,
    eps: float = PGD_EPS,
    alpha: float = PGD_ALPHA,
    steps: int = PGD_STEPS,
):
    """
    Create PGD (Projected Gradient Descent) attack.
    
    Args:
        model: Target model to attack
        eps: Maximum perturbation (L-inf norm)
        alpha: Step size for each iteration
        steps: Number of attack iterations
        
    Returns:
        torchattacks PGD attack instance
    """
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)


def create_cw_attack(
    model: nn.Module,
    c: float = CW_C,
    kappa: float = CW_KAPPA,
    steps: int = CW_STEPS,
    lr: float = CW_LR,
):
    """
    Create CW (Carlini & Wagner) L2 attack.
    
    Args:
        model: Target model to attack
        c: Weight of the classification loss
        kappa: Confidence parameter
        steps: Number of optimization steps
        lr: Learning rate for optimization
        
    Returns:
        torchattacks CW attack instance
    """
    return torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)


def run_attack(attack, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Run an attack on a batch of images.
    
    Args:
        attack: torchattacks attack instance
        images: Clean images tensor (N, C, H, W)
        labels: True labels tensor (N,)
        
    Returns:
        Adversarial images tensor (N, C, H, W)
    """
    return attack(images, labels)


def get_all_attacks(model: nn.Module) -> dict:
    """
    Create all baseline attacks for a model.
    
    Args:
        model: Target model to attack
        
    Returns:
        Dictionary with attack name -> attack instance
    """
    return {
        'FGSM': create_fgsm_attack(model),
        'PGD': create_pgd_attack(model),
        'CW': create_cw_attack(model),
    }
