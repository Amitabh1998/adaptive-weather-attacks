"""
Perceptual and realism metrics for evaluating adversarial perturbations.

Implements:
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
"""

from typing import Dict, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from ..config import DEVICE


class RealismMetrics:
    """
    Class for computing perceptual quality metrics.
    
    Computes LPIPS, SSIM, and PSNR between original and perturbed images.
    
    Args:
        device: Device to run computations on
        
    Example:
        >>> metrics = RealismMetrics()
        >>> scores = metrics.compute_all(original_images, perturbed_images)
        >>> print(f"LPIPS: {scores['lpips']:.3f}, SSIM: {scores['ssim']:.3f}")
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self._lpips_model = None
    
    @property
    def lpips_model(self):
        """Lazy loading of LPIPS model."""
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
        return self._lpips_model
    
    def compute_lpips(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).
        
        Lower LPIPS = more similar (more realistic perturbation).
        
        Args:
            images1: First batch of images [B, C, H, W], range [-1, 1] or [0, 1]
            images2: Second batch of images [B, C, H, W]
            
        Returns:
            Mean LPIPS score
        """
        images1 = images1.to(self.device)
        images2 = images2.to(self.device)
        
        # LPIPS expects images in [-1, 1] range
        if images1.min() >= 0:
            images1 = images1 * 2 - 1
            images2 = images2 * 2 - 1
        
        with torch.no_grad():
            lpips_scores = self.lpips_model(images1, images2)
        
        return lpips_scores.mean().item()
    
    def compute_ssim(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> float:
        """
        Compute SSIM (Structural Similarity Index).
        
        Higher SSIM = more similar (more realistic perturbation).
        
        Args:
            images1: First batch of images [B, C, H, W]
            images2: Second batch of images [B, C, H, W]
            
        Returns:
            Mean SSIM score
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy
        if isinstance(images1, torch.Tensor):
            images1 = images1.cpu().numpy()
        if isinstance(images2, torch.Tensor):
            images2 = images2.cpu().numpy()
        
        # Ensure correct shape [B, H, W, C]
        if images1.shape[1] == 3:  # [B, C, H, W]
            images1 = np.transpose(images1, (0, 2, 3, 1))
            images2 = np.transpose(images2, (0, 2, 3, 1))
        
        ssim_scores = []
        for img1, img2 in zip(images1, images2):
            score = ssim(
                img1, img2,
                channel_axis=2,
                data_range=img1.max() - img1.min()
            )
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    def compute_psnr(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> float:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio).
        
        Higher PSNR = less distortion.
        
        Args:
            images1: First batch of images [B, C, H, W]
            images2: Second batch of images [B, C, H, W]
            
        Returns:
            Mean PSNR score in dB
        """
        if isinstance(images1, torch.Tensor):
            images1 = images1.cpu().numpy()
        if isinstance(images2, torch.Tensor):
            images2 = images2.cpu().numpy()
        
        mse = np.mean((images1 - images2) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0 if images1.max() <= 1.0 else 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def compute_linf(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> float:
        """
        Compute L-infinity norm (maximum perturbation).
        
        Args:
            images1: First batch of images
            images2: Second batch of images
            
        Returns:
            Maximum absolute difference
        """
        if isinstance(images1, torch.Tensor):
            diff = (images1 - images2).abs()
            return diff.max().item()
        else:
            diff = np.abs(images1 - images2)
            return diff.max()
    
    def compute_l2(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> float:
        """
        Compute mean L2 norm of perturbation.
        
        Args:
            images1: First batch of images
            images2: Second batch of images
            
        Returns:
            Mean L2 norm per image
        """
        if isinstance(images1, torch.Tensor):
            diff = images1 - images2
            # L2 norm per image, then mean
            l2_per_image = diff.view(diff.size(0), -1).norm(p=2, dim=1)
            return l2_per_image.mean().item()
        else:
            diff = images1 - images2
            l2_per_image = np.linalg.norm(
                diff.reshape(diff.shape[0], -1), ord=2, axis=1
            )
            return l2_per_image.mean()
    
    def compute_all(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all realism metrics.
        
        Args:
            original: Original images
            perturbed: Perturbed images
            
        Returns:
            Dictionary with all metrics
        """
        return {
            'lpips': self.compute_lpips(original, perturbed),
            'ssim': self.compute_ssim(original, perturbed),
            'psnr': self.compute_psnr(original, perturbed),
            'linf': self.compute_linf(original, perturbed),
            'l2': self.compute_l2(original, perturbed),
        }


# Convenience functions
def compute_lpips(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: str = DEVICE,
) -> float:
    """Compute LPIPS score between two batches."""
    metrics = RealismMetrics(device)
    return metrics.compute_lpips(images1, images2)


def compute_ssim(
    images1: torch.Tensor,
    images2: torch.Tensor,
) -> float:
    """Compute SSIM score between two batches."""
    metrics = RealismMetrics()
    return metrics.compute_ssim(images1, images2)


def compute_psnr(
    images1: torch.Tensor,
    images2: torch.Tensor,
) -> float:
    """Compute PSNR between two batches."""
    metrics = RealismMetrics()
    return metrics.compute_psnr(images1, images2)
