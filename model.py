"""
FractalGen MIDI Model - Backward Compatibility Interface

This file provides backward compatibility by importing from the modular structure.
All model components are now organized in the models/ package for better readability.

For new code, prefer importing directly from models package:
    from models import PianoRollFractalGen, fractalmar_piano
"""

# Import all components from the modular structure
from models import (
    # Attention modules
    Attention,
    CausalAttention,
    # Transformer blocks
    Block,
    CausalBlock,
    # Generators
    PianoRollMAR,
    PianoRollAR,
    PianoRollVelocityLoss,
    MlmLayer,
    # Main model
    PianoRollFractalGen,
    fractalmar_piano,
    # Utilities
    mask_by_order,
    scaled_dot_product_attention,
    count_parameters,
    # Generation functions
    conditional_generation,
    inpainting_generation,
)

__all__ = [
    # Attention modules
    'Attention',
    'CausalAttention',
    # Transformer blocks
    'Block',
    'CausalBlock',
    # Generators
    'PianoRollMAR',
    'PianoRollAR',
    'PianoRollVelocityLoss',
    'MlmLayer',
    # Main model
    'PianoRollFractalGen',
    'fractalmar_piano',
    # Utilities
    'mask_by_order',
    'scaled_dot_product_attention',
    'count_parameters',
    # Generation functions
    'conditional_generation',
    'inpainting_generation',
]

