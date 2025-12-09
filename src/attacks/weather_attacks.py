"""
Weather-based adversarial attacks using diffusion models.

This module provides a simple interface for generating weather
perturbations. The Variable CFG implementation is in src/diffusion/.
"""

from typing import Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from ..config import DEVICE, WEATHER_PROMPTS, OUTPUTS_DIR


class WeatherAttack:
    """
    Weather-based adversarial attack using diffusion models.
    
    This is a high-level wrapper that uses the VariableCFGPipeline
    from src/diffusion for the actual generation.
    
    Args:
        weather_type: Type of weather ('fog', 'rain', 'snow', 'night', 'glare')
        cfg_schedule: CFG schedule type ('constant', 'linear', 'cosine', 'step')
        strength: Diffusion strength (0-1)
        
    Example:
        >>> attack = WeatherAttack('fog', cfg_schedule='linear')
        >>> adv_images = attack.generate(images)
    """
    
    def __init__(
        self,
        weather_type: str = "fog",
        cfg_schedule: str = "constant",
        strength: float = 0.5,
        device: str = DEVICE,
    ):
        self.weather_type = weather_type
        self.cfg_schedule = cfg_schedule
        self.strength = strength
        self.device = device
        
        # Will be lazily initialized
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy loading of diffusion pipeline."""
        if self._pipeline is None:
            from ..diffusion import VariableCFGPipeline
            self._pipeline = VariableCFGPipeline(device=self.device)
        return self._pipeline
    
    def get_prompt(self) -> str:
        """Get a random prompt for the weather type."""
        prompts = WEATHER_PROMPTS.get(self.weather_type, ["in adverse weather"])
        return random.choice(prompts)
    
    def generate(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Generate weather-perturbed adversarial images.
        
        Args:
            images: Batch of images [B, C, H, W]
            prompts: Optional custom prompts (one per image)
            
        Returns:
            Perturbed images [B, C, H, W]
        """
        batch_size = images.shape[0]
        
        if prompts is None:
            prompts = [self.get_prompt() for _ in range(batch_size)]
        
        adv_images = self.pipeline.generate_batch(
            images=images,
            prompts=prompts,
            cfg_schedule=self.cfg_schedule,
            strength=self.strength,
        )
        
        return adv_images
    
    def generate_single(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate weather perturbation for a single PIL image.
        
        Args:
            image: Input PIL image
            prompt: Optional custom prompt
            
        Returns:
            Perturbed PIL image
        """
        if prompt is None:
            prompt = self.get_prompt()
        
        return self.pipeline.generate_single(
            image=image,
            prompt=prompt,
            cfg_schedule=self.cfg_schedule,
            strength=self.strength,
        )
    
    def attack_dataset(
        self,
        dataset,
        num_samples: int = 100,
        save_images: bool = True,
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Attack a dataset and collect results.
        
        Args:
            dataset: Dataset with (image, label) pairs (PIL images)
            num_samples: Number of samples to attack
            save_images: Whether to save generated images
            output_dir: Directory to save images
            
        Returns:
            List of result dictionaries
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR / "adversarial" / self.weather_type
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for idx in tqdm(indices, desc=f"Generating {self.weather_type} attacks"):
            image, label = dataset[idx]
            prompt = self.get_prompt()
            
            # Generate adversarial image
            adv_image = self.generate_single(image, prompt)
            
            result = {
                'index': idx,
                'label': label,
                'prompt': prompt,
                'weather_type': self.weather_type,
                'cfg_schedule': self.cfg_schedule,
            }
            
            if save_images:
                orig_path = output_dir / f"{idx}_original.png"
                adv_path = output_dir / f"{idx}_adversarial.png"
                
                image.save(orig_path)
                adv_image.save(adv_path)
                
                result['original_path'] = str(orig_path)
                result['adversarial_path'] = str(adv_path)
            
            results.append(result)
        
        return results


# Import Path for the method above
from pathlib import Path
