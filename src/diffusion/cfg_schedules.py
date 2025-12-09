"""
Variable Classifier-Free Guidance (V-CFG) Schedule Functions.

This module implements different CFG scheduling strategies for diffusion models.
The key insight is that different stages of denoising benefit from different
guidance strengths:

- Early steps (high noise): High guidance → establish structure
- Late steps (low noise): Low guidance → natural texture details

Schedule Functions:
- constant: Traditional fixed CFG (baseline)
- linear: Linear decay from high to low
- cosine: Smooth cosine annealing
- step: Abrupt transition at midpoint
- inverse: Low → high (for comparison)

Usage:
    >>> schedule_fn = get_cfg_schedule_fn('linear', w_start=12.0, w_end=3.0)
    >>> for t in range(num_steps):
    ...     progress = t / num_steps  # 0 to 1
    ...     guidance_scale = schedule_fn(progress)
"""

from typing import Callable, Optional
from enum import Enum
import math


class CFGSchedule(str, Enum):
    """Enumeration of available CFG schedule types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    INVERSE = "inverse"


def constant_schedule(
    progress: float,
    w: float = 7.5,
    **kwargs
) -> float:
    """
    Constant CFG schedule (baseline).
    
    Args:
        progress: Denoising progress from 0 (start) to 1 (end)
        w: Fixed guidance scale
        
    Returns:
        Guidance scale (constant)
    """
    return w


def linear_schedule(
    progress: float,
    w_start: float = 12.0,
    w_end: float = 3.0,
    **kwargs
) -> float:
    """
    Linear decay CFG schedule.
    
    Linearly interpolates from w_start to w_end as denoising progresses.
    
    Args:
        progress: Denoising progress from 0 (start) to 1 (end)
        w_start: Guidance scale at start (high noise)
        w_end: Guidance scale at end (low noise)
        
    Returns:
        Interpolated guidance scale
        
    Example:
        progress=0.0 → w_start (12.0)
        progress=0.5 → 7.5 (midpoint)
        progress=1.0 → w_end (3.0)
    """
    return w_start + (w_end - w_start) * progress


def cosine_schedule(
    progress: float,
    w_start: float = 12.0,
    w_end: float = 3.0,
    **kwargs
) -> float:
    """
    Cosine annealing CFG schedule.
    
    Smooth transition using cosine function. Slower change at endpoints,
    faster change in the middle.
    
    Args:
        progress: Denoising progress from 0 (start) to 1 (end)
        w_start: Guidance scale at start
        w_end: Guidance scale at end
        
    Returns:
        Cosine-annealed guidance scale
    """
    # Cosine annealing: starts slow, speeds up, then slows down
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return w_end + (w_start - w_end) * cosine_factor


def step_schedule(
    progress: float,
    w_high: float = 12.0,
    w_low: float = 3.0,
    transition: float = 0.5,
    **kwargs
) -> float:
    """
    Step function CFG schedule.
    
    Abrupt transition from high to low guidance at a specified point.
    
    Args:
        progress: Denoising progress from 0 (start) to 1 (end)
        w_high: Guidance scale before transition
        w_low: Guidance scale after transition
        transition: Progress point for transition (0-1)
        
    Returns:
        Step function guidance scale
    """
    if progress < transition:
        return w_high
    return w_low


def inverse_schedule(
    progress: float,
    w_start: float = 3.0,
    w_end: float = 12.0,
    **kwargs
) -> float:
    """
    Inverse (increasing) CFG schedule.
    
    Opposite of linear decay - starts low and increases.
    Useful as a comparison baseline to show that decay is better.
    
    Args:
        progress: Denoising progress from 0 (start) to 1 (end)
        w_start: Guidance scale at start (low)
        w_end: Guidance scale at end (high)
        
    Returns:
        Linearly increasing guidance scale
    """
    return w_start + (w_end - w_start) * progress


# Registry of schedule functions
SCHEDULE_REGISTRY = {
    CFGSchedule.CONSTANT: constant_schedule,
    CFGSchedule.LINEAR: linear_schedule,
    CFGSchedule.COSINE: cosine_schedule,
    CFGSchedule.STEP: step_schedule,
    CFGSchedule.INVERSE: inverse_schedule,
    # Also support string keys
    "constant": constant_schedule,
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "step": step_schedule,
    "inverse": inverse_schedule,
}


def get_cfg_schedule_fn(
    schedule_type: str,
    **kwargs
) -> Callable[[float], float]:
    """
    Get a CFG schedule function with bound parameters.
    
    Args:
        schedule_type: One of 'constant', 'linear', 'cosine', 'step', 'inverse'
        **kwargs: Schedule-specific parameters
        
    Returns:
        Callable that takes progress (0-1) and returns guidance scale
        
    Example:
        >>> schedule_fn = get_cfg_schedule_fn('linear', w_start=12.0, w_end=3.0)
        >>> schedule_fn(0.0)  # Returns 12.0
        >>> schedule_fn(0.5)  # Returns 7.5
        >>> schedule_fn(1.0)  # Returns 3.0
    """
    schedule_type = schedule_type.lower()
    
    if schedule_type not in SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule: {schedule_type}. "
            f"Available: {list(SCHEDULE_REGISTRY.keys())}"
        )
    
    base_fn = SCHEDULE_REGISTRY[schedule_type]
    
    # Return a partial function with bound kwargs
    def schedule_fn(progress: float) -> float:
        return base_fn(progress, **kwargs)
    
    return schedule_fn


def visualize_schedules(
    num_steps: int = 30,
    schedules: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize CFG schedules over denoising steps.
    
    Args:
        num_steps: Number of denoising steps
        schedules: List of schedule names to visualize
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if schedules is None:
        schedules = ["constant", "linear", "cosine", "step"]
    
    progress_values = np.linspace(0, 1, num_steps)
    
    plt.figure(figsize=(10, 6))
    
    for schedule_name in schedules:
        if schedule_name == "constant":
            schedule_fn = get_cfg_schedule_fn(schedule_name, w=7.5)
        else:
            schedule_fn = get_cfg_schedule_fn(schedule_name, w_start=12.0, w_end=3.0)
        
        guidance_values = [schedule_fn(p) for p in progress_values]
        plt.plot(progress_values, guidance_values, label=schedule_name, linewidth=2)
    
    plt.xlabel("Denoising Progress", fontsize=12)
    plt.ylabel("Guidance Scale", fontsize=12)
    plt.title("Variable CFG Schedules", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Default schedule parameters
DEFAULT_PARAMS = {
    "constant": {"w": 7.5},
    "linear": {"w_start": 12.0, "w_end": 3.0},
    "cosine": {"w_start": 12.0, "w_end": 3.0},
    "step": {"w_high": 12.0, "w_low": 3.0, "transition": 0.5},
    "inverse": {"w_start": 3.0, "w_end": 12.0},
}


def get_default_schedule_fn(schedule_type: str) -> Callable[[float], float]:
    """Get a schedule function with default parameters."""
    params = DEFAULT_PARAMS.get(schedule_type.lower(), {})
    return get_cfg_schedule_fn(schedule_type, **params)
