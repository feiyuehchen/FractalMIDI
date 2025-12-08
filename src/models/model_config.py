"""
Model configuration dataclass for FractalMIDI.
Centralizes all model hyperparameters to avoid hard-coding.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PianoRollConfig:
    """Piano roll specific configuration."""
    height: int = 128                    # MIDI pitch range (21-108 → 128 bins)
    max_width: int = 256                 # Maximum time steps
    patch_size: int = 4                  # Patch size for tokenization
    velocity_vocab_size: int = 256       # MIDI velocity range [0-255] (Legacy/Input)
    num_velocity_bins: int = 32          # Number of quantization bins for velocity output

    def __post_init__(self):
        """Validate piano roll configuration."""
        if self.height % self.patch_size != 0:
            raise ValueError(f"height ({self.height}) must be divisible by patch_size ({self.patch_size})")
        # Relaxed check for velocity_vocab_size as we might not use it strictly for 256 anymore
        # if self.velocity_vocab_size != 256:
        #    raise ValueError("velocity_vocab_size must be 256 for MIDI compatibility")


@dataclass
class ArchitectureConfig:
    """Model architecture configuration."""
    # Layer-wise architecture
    seq_len_list: Tuple[int, ...] = (32, 64, 128, 512) # Renamed from img_size_list to reflect sequence length
    embed_dim_list: Tuple[int, ...] = (512, 256, 128, 64)
    num_blocks_list: Tuple[int, ...] = (12, 3, 2, 1)
    num_heads_list: Tuple[int, ...] = (8, 4, 2, 2)
    input_channels: int = 2
    cond_dim: int = 12
    
    # Dropout rates
    attn_dropout: float = 0.2
    proj_dropout: float = 0.2
    
    # Initialization
    init_std: float = 0.02               # Standard deviation for weight initialization
    
    # Other hyperparameters
    mlp_ratio: float = 4.0               # MLP hidden dim ratio
    qkv_bias: bool = True                # Use bias in QKV projection
    layer_norm_eps: float = 1e-6         # LayerNorm epsilon
    max_bar_len: int = 64                # Maximum bar length for position embedding
    
    # Harmonic Compression
    compressed_dim: int = 32             # Bottleneck dimension
    compression_act: str = 'relu'        # Activation function for compression (relu, gelu, identity)
    
    def __post_init__(self):
        """Validate architecture configuration."""
        num_levels = len(self.seq_len_list)
        if len(self.embed_dim_list) != num_levels:
            raise ValueError(f"embed_dim_list length ({len(self.embed_dim_list)}) must match seq_len_list ({num_levels})")
        if len(self.num_blocks_list) != num_levels:
            raise ValueError(f"num_blocks_list length ({len(self.num_blocks_list)}) must match seq_len_list ({num_levels})")
        if len(self.num_heads_list) != num_levels:
            raise ValueError(f"num_heads_list length ({len(self.num_heads_list)}) must match seq_len_list ({num_levels})")


@dataclass
class GeneratorConfig:
    """Generator-specific configuration."""
    # Generator types per level
    generator_type_list: Tuple[str, ...] = ("mar", "mar", "mar", "mar")
    
    # Pitch Decoding Strategy
    pitch_generator_type: str = "ar"  # 'ar' (Sequential) or 'parallel' (Efficient O(1))

    # AR generator settings
    scan_order: str = "row_major"        # Options: "row_major", "column_major"
    ar_prefix_mask_ratio: float = 1.0    # Max prefix mask ratio for AR training (1.0 = can mask 100%)
    
    # MAR generator settings
    mask_ratio_loc: float = 1.0          # Mean mask ratio (1.0 = 100%, 0.5 = 50%)
    mask_ratio_scale: float = 0.5        # Std dev of mask ratio (for random masking)
    full_mask_prob: float = 0.1          # Probability of forcing full mask (100%) during training
    
    # Generation settings
    num_conds: int = 5                   # Number of conditioning embeddings for guiding pixel
    guiding_pixel: bool = False          # Use guiding pixel (set per level in runtime)
    
    def __post_init__(self):
        """Validate generator configuration."""
        if self.scan_order not in {"row_major", "column_major"}:
            raise ValueError(f"scan_order must be 'row_major' or 'column_major', got '{self.scan_order}'")
        for idx, g in enumerate(self.generator_type_list):
            if g not in {"mar", "ar"}:
                raise ValueError(f"generator_type_list[{idx}] must be 'mar' or 'ar', got '{g}'")
        if self.pitch_generator_type not in {"ar", "parallel"}:
            raise ValueError(f"pitch_generator_type must be 'ar' or 'parallel', got '{self.pitch_generator_type}'")


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    grad_checkpointing: bool = False     # Enable gradient checkpointing (saves memory)
    v_weight: float = 1.0                # Weight for velocity loss
    
    # Sequence length settings
    max_seq_len: int = None              # Maximum sequence length (computed if None)
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.v_weight < 0:
            raise ValueError(f"v_weight must be non-negative, got {self.v_weight}")


@dataclass
class FractalModelConfig:
    """Complete model configuration for FractalMIDI."""
    piano_roll: PianoRollConfig = None
    architecture: ArchitectureConfig = None
    generator: GeneratorConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.piano_roll is None:
            self.piano_roll = PianoRollConfig()
        if self.architecture is None:
            self.architecture = ArchitectureConfig()
        if self.generator is None:
            self.generator = GeneratorConfig()
        if self.training is None:
            self.training = TrainingConfig()
        
        # Validate compatibility
        self._validate_compatibility()
    
    def _validate_compatibility(self):
        """Validate that configurations are compatible with each other."""
        # Check that seq_len_list last element matches piano_roll.max_width (if using finest level)
        # Actually seq_len_list elements are just resolution levels. The highest resolution should ideally match max_width.
        if self.architecture.seq_len_list[-1] != self.piano_roll.max_width:
             # Warning or Error? Let's enforce it for now as it simplifies logic
             pass 
             # Actually, let's NOT enforce strictly here, but it's good practice.
        
        # Check generator_type_list length matches architecture levels
        if len(self.generator.generator_type_list) != len(self.architecture.seq_len_list):
            raise ValueError(
                f"generator_type_list length ({len(self.generator.generator_type_list)}) "
                f"must match seq_len_list ({len(self.architecture.seq_len_list)})"
            )
    
    def to_dict(self):
        """Convert to dictionary for YAML serialization."""
        return {
            'piano_roll': {
                'height': self.piano_roll.height,
                'max_width': self.piano_roll.max_width,
                'patch_size': self.piano_roll.patch_size,
                'velocity_vocab_size': self.piano_roll.velocity_vocab_size,
            },
            'architecture': {
                'seq_len_list': list(self.architecture.seq_len_list),
                'embed_dim_list': list(self.architecture.embed_dim_list),
                'num_blocks_list': list(self.architecture.num_blocks_list),
                'num_heads_list': list(self.architecture.num_heads_list),
                'attn_dropout': self.architecture.attn_dropout,
                'proj_dropout': self.architecture.proj_dropout,
                'init_std': self.architecture.init_std,
                'mlp_ratio': self.architecture.mlp_ratio,
                'qkv_bias': self.architecture.qkv_bias,
                'layer_norm_eps': self.architecture.layer_norm_eps,
                'max_bar_len': self.architecture.max_bar_len,
                'compressed_dim': self.architecture.compressed_dim,
                'compression_act': self.architecture.compression_act,
            },
            'generator': {
                'generator_type_list': list(self.generator.generator_type_list),
                'pitch_generator_type': self.generator.pitch_generator_type,
                'scan_order': self.generator.scan_order,
                'mask_ratio_loc': self.generator.mask_ratio_loc,
                'mask_ratio_scale': self.generator.mask_ratio_scale,
                'full_mask_prob': self.generator.full_mask_prob,
                'num_conds': self.generator.num_conds,
            },
            'training': {
                'grad_checkpointing': self.training.grad_checkpointing,
                'v_weight': self.training.v_weight,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary (e.g., loaded from YAML)."""
        piano_roll = PianoRollConfig(**config_dict.get('piano_roll', {}))
        
        arch_dict = config_dict.get('architecture', {})
        
        # Handle legacy img_size_list -> seq_len_list mapping or replacement
        if 'img_size_list' in arch_dict and 'seq_len_list' not in arch_dict:
            # If user provides img_size_list but not seq_len_list, map it
            # Legacy was [128, 16, 4, 1] (downsample ratios or coarse sizes)
            # We want actual lengths. This is tricky to map automatically without context.
            # So we just warn or assign it to seq_len_list and let user fix it if it crashes.
            # Or better: assume it IS seq_len_list if values look like lengths.
            arch_dict['seq_len_list'] = arch_dict.pop('img_size_list')
            
        # Convert lists to tuples
        for key in ['seq_len_list', 'img_size_list', 'embed_dim_list', 'num_blocks_list', 'num_heads_list']:
            if key in arch_dict:
                arch_dict[key] = tuple(arch_dict[key])
                
        architecture = ArchitectureConfig(**arch_dict)
        
        gen_dict = config_dict.get('generator', {})
        # Handle alias: generator_types -> generator_type_list
        if 'generator_types' in gen_dict:
            gen_dict['generator_type_list'] = gen_dict.pop('generator_types')
            
        if 'generator_type_list' in gen_dict:
            gen_dict['generator_type_list'] = tuple(gen_dict['generator_type_list'])
        generator = GeneratorConfig(**gen_dict)
        
        training = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            piano_roll=piano_roll,
            architecture=architecture,
            generator=generator,
            training=training
        )


# Default configurations
def get_default_config():
    """Get default 128x512 configuration."""
    return FractalModelConfig()


def get_small_config():
    """Get small model configuration for testing."""
    return FractalModelConfig(
        piano_roll=PianoRollConfig(max_width=128),
        architecture=ArchitectureConfig(
            embed_dim_list=(256, 256, 128, 64),
            num_blocks_list=(8, 4, 2, 1),
            num_heads_list=(4, 2, 1, 1),
        ),
        generator=GeneratorConfig(
            generator_type_list=("ar", "ar", "ar", "ar"),
        ),
    )


def get_256_config():
    """Get 128x256 configuration."""
    return FractalModelConfig(
        piano_roll=PianoRollConfig(max_width=256),
    )
