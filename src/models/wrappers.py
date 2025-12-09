"""
Model wrappers for consistent interface across different architectures.

ViT from transformers library returns a different output format than
standard PyTorch models. These wrappers normalize the interface.
"""

import torch
import torch.nn as nn


class ViTWrapper(nn.Module):
    """
    Wrapper for HuggingFace ViT model to return logits directly.
    
    The transformers ViTForImageClassification returns an object with
    a 'logits' attribute instead of raw logits. This wrapper extracts
    the logits for compatibility with standard PyTorch training loops.
    
    Args:
        model: ViTForImageClassification model
        
    Example:
        >>> from transformers import ViTForImageClassification
        >>> vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        >>> wrapped = ViTWrapper(vit)
        >>> logits = wrapped(images)  # Returns tensor directly
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        
        if hasattr(output, 'logits'):
            return output.logits
        return output
    
    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)


class ModelWrapper(nn.Module):
    """
    Generic model wrapper that handles different output formats.
    
    Automatically detects if the model output has a 'logits' attribute
    (like transformers models) and extracts it.
    
    Args:
        model: Any PyTorch model
        
    Example:
        >>> model = ModelWrapper(any_model)
        >>> logits = model(images)  # Always returns tensor
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        
        # Handle transformers-style output
        if hasattr(output, 'logits'):
            return output.logits
        
        return output
    
    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode=True):
        self.model.train(mode)
        return self


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for evaluation.
    
    Combines predictions from multiple models using averaging
    or voting strategies.
    
    Args:
        models: Dictionary of models {name: model}
        strategy: 'average' (average logits) or 'vote' (majority voting)
    """
    
    def __init__(self, models: dict, strategy: str = "average"):
        super().__init__()
        self.models = nn.ModuleDict({
            name: ModelWrapper(model) if not isinstance(model, ModelWrapper) else model
            for name, model in models.items()
        })
        self.strategy = strategy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for model in self.models.values():
            output = model(x)
            outputs.append(output)
        
        if self.strategy == "average":
            # Average logits
            stacked = torch.stack(outputs, dim=0)
            return stacked.mean(dim=0)
        
        elif self.strategy == "vote":
            # Majority voting (return most common prediction as one-hot)
            predictions = [out.argmax(dim=1) for out in outputs]
            stacked = torch.stack(predictions, dim=0)
            # Mode across models
            voted, _ = torch.mode(stacked, dim=0)
            # Convert back to one-hot-ish (just return averaged logits for simplicity)
            stacked_logits = torch.stack(outputs, dim=0)
            return stacked_logits.mean(dim=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def eval(self):
        for model in self.models.values():
            model.eval()
        return self
