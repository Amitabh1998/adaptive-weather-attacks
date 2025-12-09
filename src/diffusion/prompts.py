"""
Weather prompt definitions for diffusion-based attacks.

Contains categorized prompts for different weather conditions
that can be used to generate adversarial perturbations.
"""

import random
from typing import List, Optional


WEATHER_PROMPTS = {
    # Fog conditions
    "fog": [
        "a traffic sign in dense fog",
        "a traffic sign in thick fog",
        "a traffic sign in heavy fog with low visibility",
        "a traffic sign barely visible through fog",
        "a traffic sign in misty conditions",
    ],
    
    # Rain conditions
    "rain": [
        "a traffic sign in heavy rain",
        "a traffic sign in rainstorm with wet reflections",
        "a traffic sign in rain at dusk",
        "a traffic sign with rain droplets on surface",
        "a traffic sign in downpour",
        "a traffic sign in rain with puddles below",
    ],
    
    # Snow conditions
    "snow": [
        "a traffic sign in thick snow",
        "a traffic sign in blizzard conditions",
        "a traffic sign covered in snow",
        "a traffic sign in snowstorm",
        "a traffic sign with snow accumulation",
        "a traffic sign in whiteout conditions",
    ],
    
    # Night/low light conditions
    "night": [
        "a traffic sign at night",
        "a traffic sign at night under streetlights",
        "a traffic sign illuminated by headlights at night",
        "a traffic sign in darkness",
        "a traffic sign at dusk in dim lighting",
        "a traffic sign at night with limited visibility",
    ],
    
    # Glare/bright light conditions
    "glare": [
        "a traffic sign in harsh sunlight with glare",
        "a traffic sign with lens flare",
        "a traffic sign in bright sunlight, overexposed",
        "a traffic sign with sun behind it, backlit",
        "a traffic sign in harsh midday sun",
        "a traffic sign with strong glare and shadows",
    ],
    
    # Combined/complex conditions
    "combined": [
        "a traffic sign in fog at night",
        "a traffic sign in rain at dusk",
        "a traffic sign in fog and snow",
        "a traffic sign at night with heavy snow",
        "a traffic sign in rainstorm at dusk",
        "a traffic sign in fog and rain",
        "a traffic sign at night in fog and rain",
    ],
    
    # Surface/visibility degradation
    "degraded": [
        "a traffic sign covered in frost",
        "a traffic sign covered in ice",
        "a traffic sign with dirty, muddy surface",
        "a traffic sign partially obscured by water droplets",
        "a faded, weathered traffic sign",
        "an old, rusty traffic sign",
    ],
}


# Flat list of all prompts
ALL_PROMPTS = [
    prompt
    for category_prompts in WEATHER_PROMPTS.values()
    for prompt in category_prompts
]


def get_weather_prompt(
    weather_type: str,
    random_select: bool = True,
) -> str:
    """
    Get a weather prompt for a specific condition.
    
    Args:
        weather_type: Type of weather ('fog', 'rain', 'snow', etc.)
        random_select: If True, randomly select from available prompts
        
    Returns:
        Weather condition prompt string
    """
    if weather_type not in WEATHER_PROMPTS:
        available = list(WEATHER_PROMPTS.keys())
        raise ValueError(
            f"Unknown weather type: {weather_type}. "
            f"Available: {available}"
        )
    
    prompts = WEATHER_PROMPTS[weather_type]
    
    if random_select:
        return random.choice(prompts)
    return prompts[0]


def get_all_prompts(weather_type: Optional[str] = None) -> List[str]:
    """
    Get all prompts, optionally filtered by weather type.
    
    Args:
        weather_type: Optional filter for specific weather type
        
    Returns:
        List of prompt strings
    """
    if weather_type is None:
        return ALL_PROMPTS.copy()
    
    if weather_type not in WEATHER_PROMPTS:
        available = list(WEATHER_PROMPTS.keys())
        raise ValueError(
            f"Unknown weather type: {weather_type}. "
            f"Available: {available}"
        )
    
    return WEATHER_PROMPTS[weather_type].copy()


def get_random_prompt() -> str:
    """Get a random prompt from any category."""
    return random.choice(ALL_PROMPTS)


def get_weather_categories() -> List[str]:
    """Get list of available weather categories."""
    return list(WEATHER_PROMPTS.keys())


# Simplified prompts for experiments (one per category)
SIMPLE_PROMPTS = {
    "fog": "a traffic sign in dense fog",
    "rain": "a traffic sign in heavy rain",
    "snow": "a traffic sign in thick snow",
    "night": "a traffic sign at night",
    "glare": "a traffic sign in harsh sunlight with glare",
}


def get_simple_prompt(weather_type: str) -> str:
    """Get the canonical simple prompt for a weather type."""
    if weather_type not in SIMPLE_PROMPTS:
        raise ValueError(f"Unknown weather type: {weather_type}")
    return SIMPLE_PROMPTS[weather_type]
