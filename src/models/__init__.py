"""
FractalMIDI Models Package

This package contains the hierarchical FractalGen model implementation
for piano roll generation, split into modular components for better readability.
"""

from .attention import Attention, CausalAttention
from .blocks import Block, CausalBlock
from .mar_generator import PianoRollMAR
from .ar_generator import PianoRollAR
from .velocity_loss import PianoRollVelocityLoss, MlmLayer
from .fractal_gen import PianoRollFractalGen, fractalmar_piano
from .utils import mask_by_order, scaled_dot_product_attention, count_parameters
from .generation import conditional_generation, inpainting_generation

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

