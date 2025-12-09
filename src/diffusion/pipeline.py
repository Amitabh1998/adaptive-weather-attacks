"""
Variable CFG Pipeline for Stable Diffusion.

This module implements a custom img2img pipeline that supports
variable guidance schedules during the denoising process.

The key modification is in the denoising loop where we compute
a different guidance scale at each timestep based on the schedule.
"""

from typing import Optional, List, Callable, Union, Dict, Any
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from .cfg_schedules import get_cfg_schedule_fn, get_default_schedule_fn
from ..config import (
    DEVICE,
    DIFFUSION_MODEL_ID,
    NUM_INFERENCE_STEPS,
    DIFFUSION_STRENGTH,
)


class VariableCFGPipeline:
    """
    Stable Diffusion img2img pipeline with Variable CFG support.
    
    This pipeline allows using different guidance scales at different
    denoising steps, enabling more realistic weather perturbations.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to run on
        dtype: Tensor dtype (float16 recommended for speed)
        
    Example:
        >>> pipeline = VariableCFGPipeline()
        >>> result = pipeline.generate_single(
        ...     image=pil_image,
        ...     prompt="a traffic sign in dense fog",
        ...     cfg_schedule="linear"
        ... )
    """
    
    def __init__(
        self,
        model_id: str = DIFFUSION_MODEL_ID,
        device: str = DEVICE,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        # Load the pipeline
        print(f"Loading Stable Diffusion from {model_id}...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable for speed
        ).to(device)
        
        # Use DDIM scheduler for deterministic sampling
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Disable progress bar for batch processing
        self.pipe.set_progress_bar_config(disable=True)
        
        print(f"✓ Pipeline loaded on {device}")
    
    def generate_single(
        self,
        image: Image.Image,
        prompt: str,
        cfg_schedule: str = "constant",
        strength: float = DIFFUSION_STRENGTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        cfg_params: Optional[Dict] = None,
    ) -> Image.Image:
        """
        Generate a single weather-perturbed image with Variable CFG.
        
        Args:
            image: Input PIL image
            prompt: Weather condition prompt
            cfg_schedule: Schedule type ('constant', 'linear', 'cosine', 'step')
            strength: Diffusion strength (0-1)
            num_inference_steps: Number of denoising steps
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            cfg_params: Optional parameters for CFG schedule
            
        Returns:
            Generated PIL image
        """
        # Get schedule function
        if cfg_params is None:
            schedule_fn = get_default_schedule_fn(cfg_schedule)
        else:
            schedule_fn = get_cfg_schedule_fn(cfg_schedule, **cfg_params)
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Ensure image is RGB and reasonable size
        image = image.convert("RGB")
        image = self._resize_image(image)
        
        # Generate with variable CFG
        result = self._generate_with_variable_cfg(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            schedule_fn=schedule_fn,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        
        return result
    
    def generate_batch(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        cfg_schedule: str = "constant",
        strength: float = DIFFUSION_STRENGTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        show_progress: bool = True,
    ) -> List[Image.Image]:
        """
        Generate weather perturbations for a batch of images.
        
        Args:
            images: List of PIL images or tensor batch
            prompts: List of prompts (one per image)
            cfg_schedule: CFG schedule type
            strength: Diffusion strength
            num_inference_steps: Number of denoising steps
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated PIL images
        """
        results = []
        
        iterator = zip(images, prompts)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Generating ({cfg_schedule} CFG)")
        
        for image, prompt in iterator:
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            
            result = self.generate_single(
                image=image,
                prompt=prompt,
                cfg_schedule=cfg_schedule,
                strength=strength,
                num_inference_steps=num_inference_steps,
            )
            results.append(result)
        
        return results
    
    def _generate_with_variable_cfg(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        schedule_fn: Callable[[float], float],
        strength: float,
        num_inference_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Core generation loop with variable CFG.
        
        This manually implements the denoising loop to allow
        changing the guidance scale at each step.
        """
        # Encode prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt)
        
        # Prepare image latents
        init_latents = self._encode_image(image)
        
        # Set up timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps, strength
        )
        
        # Add noise to latents
        noise = torch.randn(
            init_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        latents = self.pipe.scheduler.add_noise(
            init_latents, noise, timesteps[:1]
        )
        
        # Track CFG values for analysis
        cfg_values = []
        
        # Denoising loop with variable CFG
        for i, t in enumerate(timesteps):
            # Calculate progress (0 at start, 1 at end)
            progress = i / len(timesteps)
            
            # Get guidance scale for this step
            guidance_scale = schedule_fn(progress)
            cfg_values.append(guidance_scale)
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            # Perform guidance with variable scale
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # Compute previous latents
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample
        
        # Decode latents to image
        image = self._decode_latents(latents)
        
        return image
    
    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        # Tokenize
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_inputs = self.pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
            uncond_embeddings = self.pipe.text_encoder(
                uncond_inputs.input_ids.to(self.device)
            )[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to latent space."""
        # Preprocess
        image = self.pipe.image_processor.preprocess(image)
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Encode
        with torch.no_grad():
            latents = self.pipe.vae.encode(image).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL image."""
        latents = latents / self.pipe.vae.config.scaling_factor
        
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)
    
    def _get_timesteps(
        self,
        num_inference_steps: int,
        strength: float,
    ):
        """Get timesteps based on strength."""
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps
        )
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.pipe.scheduler.timesteps[t_start:]
        
        return timesteps, num_inference_steps - t_start
    
    def _resize_image(
        self,
        image: Image.Image,
        size: int = 512,
    ) -> Image.Image:
        """Resize image to appropriate size for SD."""
        w, h = image.size
        
        # Resize to at least 512x512, maintaining aspect ratio
        if w < size or h < size:
            scale = size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Make dimensions divisible by 8
        w, h = image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        return image
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert normalized tensor to PIL image."""
        from ..data.transforms import tensor_to_pil
        return tensor_to_pil(tensor)
    
    def compare_schedules(
        self,
        image: Image.Image,
        prompt: str,
        schedules: List[str] = ["constant", "linear", "cosine"],
        strength: float = DIFFUSION_STRENGTH,
        seed: int = 42,
    ) -> Dict[str, Image.Image]:
        """
        Generate images with different CFG schedules for comparison.
        
        Args:
            image: Input image
            prompt: Weather prompt
            schedules: List of schedule types to compare
            strength: Diffusion strength
            seed: Random seed (same for all for fair comparison)
            
        Returns:
            Dictionary mapping schedule name to generated image
        """
        results = {"original": image}
        
        for schedule in schedules:
            result = self.generate_single(
                image=image,
                prompt=prompt,
                cfg_schedule=schedule,
                strength=strength,
                seed=seed,
            )
            results[schedule] = result
        
        return results
