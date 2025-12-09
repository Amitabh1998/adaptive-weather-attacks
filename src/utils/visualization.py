"""
Visualization utilities for experiments and results.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from ..config import FIGURES_DIR, MEAN, STD


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable numpy array."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().detach().cpu()
    
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to [H, W, C]
    return tensor.permute(1, 2, 0).numpy()


def plot_images(
    images: Union[List[Image.Image], List[torch.Tensor], Dict[str, Image.Image]],
    titles: Optional[List[str]] = None,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a row of images.
    
    Args:
        images: List of images or dict mapping title to image
        titles: Optional titles for each image
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            img = denormalize_tensor(img)
        elif isinstance(img, Image.Image):
            img = np.array(img) / 255.0
        
        ax.imshow(img)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_attack_comparison(
    original: Union[Image.Image, torch.Tensor],
    attacked_images: Dict[str, Union[Image.Image, torch.Tensor]],
    predictions: Optional[Dict[str, int]] = None,
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot original vs. multiple attack results.
    
    Args:
        original: Original image
        attacked_images: Dict mapping attack name to attacked image
        predictions: Optional dict mapping attack name to prediction
        true_label: True class label
        save_path: Optional path to save figure
    """
    n = len(attacked_images) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    # Plot original
    if isinstance(original, torch.Tensor):
        original = denormalize_tensor(original)
    elif isinstance(original, Image.Image):
        original = np.array(original) / 255.0
    
    axes[0].imshow(original)
    axes[0].set_title("Original" + (f"\nLabel: {true_label}" if true_label else ""))
    axes[0].axis('off')
    
    # Plot attacks
    for i, (name, img) in enumerate(attacked_images.items(), 1):
        if isinstance(img, torch.Tensor):
            img = denormalize_tensor(img)
        elif isinstance(img, Image.Image):
            img = np.array(img) / 255.0
        
        axes[i].imshow(img)
        
        title = name
        if predictions and name in predictions:
            pred = predictions[name]
            correct = "✓" if pred == true_label else "✗"
            title += f"\nPred: {pred} {correct}"
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_cfg_schedules(
    num_steps: int = 30,
    schedules: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize CFG schedules over denoising steps.
    
    Args:
        num_steps: Number of denoising steps
        schedules: List of schedule types to plot
        save_path: Optional path to save figure
    """
    from ..diffusion.cfg_schedules import get_default_schedule_fn
    
    if schedules is None:
        schedules = ["constant", "linear", "cosine", "step"]
    
    progress = np.linspace(0, 1, num_steps)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedules)))
    
    for schedule, color in zip(schedules, colors):
        schedule_fn = get_default_schedule_fn(schedule)
        values = [schedule_fn(p) for p in progress]
        ax.plot(progress, values, label=schedule.capitalize(), 
                linewidth=2.5, color=color)
    
    ax.set_xlabel("Denoising Progress", fontsize=12)
    ax.set_ylabel("Guidance Scale", fontsize=12)
    ax.set_title("Variable CFG Schedules", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_results_table(
    results: Dict[str, Dict],
    metrics: List[str] = ['asr', 'lpips', 'ssim'],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot results as a formatted table/heatmap.
    
    Args:
        results: Nested dict {method: {metric: value}}
        metrics: Metrics to include
        save_path: Optional path to save figure
    """
    methods = list(results.keys())
    data = []
    
    for method in methods:
        row = []
        for metric in metrics:
            value = results[method].get(metric, 0)
            row.append(value)
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(8, len(methods) * 0.5 + 2))
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        xticklabels=metrics,
        yticklabels=methods,
        cmap='RdYlGn',
        ax=ax
    )
    
    ax.set_title("Experiment Results", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_transfer_matrix(
    transfer_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot transferability matrix as heatmap.
    
    Args:
        transfer_results: Nested dict {source: {target: asr}}
        save_path: Optional path to save figure
    """
    sources = list(transfer_results.keys())
    targets = sources  # Assume same models
    
    matrix = np.zeros((len(sources), len(targets)))
    
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            matrix[i, j] = transfer_results[source].get(target, 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        xticklabels=targets,
        yticklabels=sources,
        cmap='YlOrRd',
        ax=ax,
        vmin=0,
        vmax=100,
    )
    
    ax.set_xlabel("Target Model", fontsize=12)
    ax.set_ylabel("Source Model", fontsize=12)
    ax.set_title("Attack Transferability (ASR %)", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 150,
) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        path: Save path
        dpi: Resolution
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved figure to {path}")
