"""
Configuration for FractalMIDI Web Application Backend.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = False  # Set to True for development
    workers: int = 1
    log_level: str = "info"


@dataclass
class ModelConfig:
    """Model configuration."""
    default_checkpoint: Optional[str] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("experiments"))
    device: str = "cuda"  # or "cpu"
    max_batch_size: int = 4
    
    # Generation defaults
    default_generator_type: str = "mar"  # "mar" or "ar"
    default_scan_order: str = "row_major"  # "row_major" or "column_major"
    default_num_iter_list: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    default_cfg: float = 1.0
    default_temperature: float = 1.0
    default_filter_threshold: float = 0.0


@dataclass
class GenerationConfig:
    """Generation configuration."""
    max_length: int = 512  # Maximum generation length in time steps
    min_length: int = 128
    default_length: int = 256
    
    # Conditional generation
    min_condition_length: int = 16
    max_condition_length: int = 128
    default_condition_length: int = 32
    
    # Inpainting
    max_inpaint_regions: int = 10
    
    # Visualization
    create_gif: bool = True
    gif_fps: int = 24
    gif_quality: int = 90
    show_progress: bool = True
    show_grid: bool = False


@dataclass
class ExampleConfig:
    """Validation examples configuration."""
    examples_dir: Path = field(default_factory=lambda: Path("data/splits/validation_examples"))
    max_examples: int = 100
    thumbnail_size: tuple = (256, 128)  # (width, height)


@dataclass
class CORSConfig:
    """CORS configuration."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Main application configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    examples: ExampleConfig = field(default_factory=ExampleConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "frontend" / "static")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "frontend" / "templates")
    
    # Security
    max_concurrent_generations: int = 5
    generation_timeout: int = 300  # seconds
    
    # Logging
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_file: str = "fractal_midi_web.log"


# Global config instance
config = AppConfig()


def load_config(config_file: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_file: Optional path to config file (JSON or YAML)
        
    Returns:
        AppConfig instance
    """
    # TODO: Implement config file loading if needed
    # For now, use defaults
    return config


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

