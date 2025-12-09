"""
Checkpoint and results saving/loading utilities.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn

from ..config import CHECKPOINT_DIR, RESULTS_DIR


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Save path
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle wrapped models
    if hasattr(model, 'model'):
        state_dict = model.model.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        'model_state_dict': state_dict,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        path: Checkpoint path
        optimizer: Optional optimizer to load state
        device: Device to load on
        
    Returns:
        Checkpoint dictionary with any extra info
    """
    path = Path(path)
    
    checkpoint = torch.load(path, map_location=device)
    
    # Handle wrapped models
    if hasattr(model, 'model'):
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded checkpoint from {path}")
    
    return checkpoint


def save_results(
    results: Dict[str, Any],
    name: str,
    format: str = 'json',
) -> Path:
    """
    Save experiment results.
    
    Args:
        results: Results dictionary
        name: Experiment name
        format: 'json' or 'pickle'
        
    Returns:
        Path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        path = RESULTS_DIR / f"{name}.json"
        
        # Convert numpy arrays to lists for JSON
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format == 'pickle':
        path = RESULTS_DIR / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"✓ Saved results to {path}")
    return path


def load_results(
    name: str,
    format: str = 'json',
) -> Dict[str, Any]:
    """
    Load experiment results.
    
    Args:
        name: Experiment name
        format: 'json' or 'pickle'
        
    Returns:
        Results dictionary
    """
    if format == 'json':
        path = RESULTS_DIR / f"{name}.json"
        with open(path, 'r') as f:
            return json.load(f)
    
    elif format == 'pickle':
        path = RESULTS_DIR / f"{name}.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def list_checkpoints() -> list:
    """List all available checkpoints."""
    checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
    return [c.stem for c in checkpoints]


def list_results() -> list:
    """List all saved results."""
    results = list(RESULTS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*.pkl"))
    return [r.stem for r in results]
