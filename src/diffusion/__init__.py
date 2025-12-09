"""
Diffusion model utilities and Variable CFG implementation.

This is the core contribution of the project.
"""

from .cfg_schedules import (
    CFGSchedule,
    get_cfg_schedule_fn,
    constant_schedule,
    linear_schedule,
    cosine_schedule,
    step_schedule,
    inverse_schedule,
)
from .pipeline import VariableCFGPipeline
from .prompts import WEATHER_PROMPTS, get_weather_prompt, get_all_prompts

__all__ = [
    # CFG Schedules
    "CFGSchedule",
    "get_cfg_schedule_fn",
    "constant_schedule",
    "linear_schedule",
    "cosine_schedule",
    "step_schedule",
    "inverse_schedule",
    # Pipeline
    "VariableCFGPipeline",
    # Prompts
    "WEATHER_PROMPTS",
    "get_weather_prompt",
    "get_all_prompts",
]
