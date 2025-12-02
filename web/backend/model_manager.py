"""
Model Manager for loading and managing FractalMIDI models.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
import yaml
import shutil

# Add parent directory to path to import FractalMIDI modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import ALL classes that might be in the YAML to ensure UnsafeLoader can find them
from trainer import FractalMIDILightningModule, FractalTrainerConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from models import PianoRollFractalGen

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    name: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    val_loss: Optional[float] = None
    generator_types: Optional[List[str]] = None
    scan_order: Optional[str] = None
    file_size_mb: Optional[float] = None


class ModelManager:
    """
    Manages loading and switching between FractalMIDI model checkpoints.
    """
    
    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        """
        Initialize model manager.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            device: Device to load models on ("cuda" or "cpu")
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.current_model: Optional[PianoRollFractalGen] = None
        self.current_checkpoint: Optional[CheckpointInfo] = None
        self.available_checkpoints: Dict[str, CheckpointInfo] = {}
        
        # Scan for available checkpoints
        self._scan_checkpoints()
        
    def _scan_checkpoints(self):
        """Scan checkpoint directory for available checkpoints."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
            return
        
        # Find all .ckpt files
        for ckpt_path in self.checkpoint_dir.rglob("*.ckpt"):
            try:
                # Determine a display name based on path
                # e.g. outputs/fractalgen_small/checkpoints/version_9/step_...
                # We want: "fractalgen_small/v9/step_..."
                
                relative_path = ckpt_path.relative_to(self.checkpoint_dir)
                parts = relative_path.parts
                
                # Try to construct a meaningful name
                # Filter out "checkpoints" from the name if present
                name_parts = [p for p in parts[:-1] if p != "checkpoints"]
                # Also normalize "version_X" to "vX" for brevity if desired, or keep as is
                # User reported issue with switching, ensure consistency
                
                name_prefix = "/".join(name_parts)
                
                filename = ckpt_path.stem # e.g. step_0010000...
                
                # Extract nice step/epoch info for short name
                short_name = filename
                if "step" in filename:
                    step_str = filename.split("-")[0] # step_0010000
                    short_name = step_str
                
                display_name = f"{name_prefix}/{short_name}" if name_prefix else short_name
                
                # Extract info from filename (e.g., "step=10000.ckpt")
                name = display_name # Use the constructed path as the key/name
                step = None
                epoch = None
                
                # Try to extract step number
                if "step" in filename:
                    try:
                        # Handle step=XXXX or step_XXXX
                        if "step=" in filename:
                            step = int(filename.split("step=")[1].split("-")[0].split(".")[0])
                        elif "step_" in filename:
                            step = int(filename.split("step_")[1].split("-")[0].split(".")[0])
                    except:
                        pass
                
                # Try to extract epoch number
                if "epoch" in filename:
                    try:
                        if "epoch=" in filename:
                            epoch = int(filename.split("epoch=")[1].split("-")[0].split(".")[0])
                        elif "epoch_" in filename:
                            epoch = int(filename.split("epoch_")[1].split("-")[0].split(".")[0])
                    except:
                        pass
                
                # Get file size
                file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                
                # Try to load config.yaml
                generator_types = None
                scan_order = None
                
                # Look for config in multiple places
                config_locations = [
                    ckpt_path.parent / "config.yaml",  # checkpoints/version_X/config.yaml
                    ckpt_path.parent.parent / "config.yaml",  # checkpoints/config.yaml
                    # Try logs/version_X/hparams.yaml
                ]
                
                # Also try to find corresponding log dir
                # Assuming structure outputs/NAME/checkpoints/version_X -> outputs/NAME/logs/version_X
                if "checkpoints" in parts:
                    ckpt_idx = parts.index("checkpoints")
                    if len(parts) > ckpt_idx + 1: # has version subdirectory
                        version_dir = parts[ckpt_idx + 1]
                        base_output_dir = Path(*parts[:ckpt_idx])
                        log_config_path = self.checkpoint_dir / base_output_dir / "logs" / version_dir / "hparams.yaml"
                        config_locations.append(log_config_path)
                
                config_data = None
                for conf_path in config_locations:
                    if conf_path.exists():
                        try:
                            with open(conf_path, 'r') as f:
                                # Use UnsafeLoader to handle python objects correctly
                                # This relies on the classes being imported above
                                if hasattr(yaml, 'unsafe_load'):
                                    config_data = yaml.unsafe_load(f)
                                else:
                                    config_data = yaml.load(f, Loader=yaml.UnsafeLoader)
                            if config_data:
                                break
                        except Exception as e:
                            logger.warning(f"Failed to load config from {conf_path}: {e}")
                            continue
                            
                if config_data:
                    # Check for direct model config or nested in hparams
                    model_conf = None
                    
                    # Helper to get attribute or dict item
                    def get_val(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key)
                        return getattr(obj, key, None)

                    if isinstance(config_data, dict) and 'model' in config_data:
                         model_conf = config_data['model']
                    elif hasattr(config_data, 'model'): # Object access
                         model_conf = config_data.model
                    else:
                         # Try 'config' -> 'model'
                         # config_data might be the object itself or a dict containing it
                         conf_obj = get_val(config_data, 'config')
                         if conf_obj:
                             model_conf = get_val(conf_obj, 'model')

                    # Sometimes hparams.yaml flattens things?
                    # Let's check if we can find generator_type_list
                    if model_conf:
                        # logger.info(f"Debug model_conf keys: {model_conf.keys()}")
                        gen_list = get_val(model_conf, 'generator_type_list')
                        if gen_list:
                            # Ensure it's a list/tuple of strings
                            # logger.info(f"Debug gen_list found: {gen_list} type: {type(gen_list)}")
                            if isinstance(gen_list, (list, tuple)):
                                generator_types = list(gen_list)
                            
                        scan_o = get_val(model_conf, 'scan_order')
                        if scan_o:
                            scan_order = scan_o
                    
                    # Fallback/Hack for version_9 which seems to have parsing issues
                    if "version_9" in name and generator_types is None:
                        generator_types = ["mar", "mar", "mar", "mar"]
                        logger.info(f"Applied version_9 fallback for {name}")
                
                info = CheckpointInfo(
                    path=ckpt_path,
                    name=name,
                    step=step,
                    epoch=epoch,
                    file_size_mb=file_size_mb,
                    generator_types=generator_types,
                    scan_order=scan_order
                )
                
                self.available_checkpoints[name] = info
                model_type = "Unknown"
                if generator_types:
                    if all(g == "mar" for g in generator_types):
                        model_type = "MAR (Diff)"
                    elif all(g == "ar" for g in generator_types):
                        model_type = "AR"
                    else:
                        model_type = f"Mixed {generator_types}"
                        
                logger.info(f"Found checkpoint: {name} ({file_size_mb:.1f} MB) - {model_type}")
                
            except Exception as e:
                logger.error(f"Error scanning checkpoint {ckpt_path}: {e}")
        
        logger.info(f"Found {len(self.available_checkpoints)} checkpoints")
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """
        List all available checkpoints.
        
        Returns:
            List of CheckpointInfo objects
        """
        return list(self.available_checkpoints.values())
    
    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> PianoRollFractalGen:
        """
        Load a checkpoint by name.
        """
        # If no name provided, use latest checkpoint
        if checkpoint_name is None:
            if not self.available_checkpoints:
                raise ValueError("No checkpoints available")
            
            # Sort by step number (if available) or name
            sorted_ckpts = sorted(
                self.available_checkpoints.values(),
                key=lambda x: x.step if x.step is not None else 0,
                reverse=True
            )
            checkpoint_info = sorted_ckpts[0]
            logger.info(f"No checkpoint specified, using latest: {checkpoint_info.name}")
        else:
            if checkpoint_name not in self.available_checkpoints:
                raise ValueError(f"Checkpoint not found: {checkpoint_name}")
            checkpoint_info = self.available_checkpoints[checkpoint_name]
        
        # Check if already loaded
        if self.current_checkpoint and self.current_checkpoint.path == checkpoint_info.path:
            if self.current_model is not None:
                logger.info(f"Checkpoint already loaded: {checkpoint_info.name}")
                return self.current_model
        
        logger.info(f"Loading checkpoint: {checkpoint_info.path}")
        
        try:
            # Load using PyTorch Lightning with strict=False to handle missing mask_token or other discrepancies
            model = FractalMIDILightningModule.load_from_checkpoint(
                str(checkpoint_info.path),
                map_location=self.device,
                strict=False
            )
            
            # Extract the actual model
            model = model.model
            model.eval()
            model.to(self.device)
            
            self.current_model = model
            self.current_checkpoint = checkpoint_info
            
            logger.info(f"Successfully loaded checkpoint: {checkpoint_info.name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def export_model(self, checkpoint_name: str, export_name: str):
        """
        Export a clean model checkpoint (no optimizer states) and config.
        
        Args:
            checkpoint_name: Name of the checkpoint to load.
            export_name: Name for the exported folder.
        """
        if checkpoint_name not in self.available_checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")
        
        ckpt_info = self.available_checkpoints[checkpoint_name]
        
        # Load the model to ensure it's valid
        # We use pl load_from_checkpoint to get hparams
        pl_module = FractalMIDILightningModule.load_from_checkpoint(
            str(ckpt_info.path),
            map_location="cpu", # Load to CPU for export
            strict=False
        )
        
        # Prepare export directory
        # Defaults to outputs/exported_models/{export_name}
        # Assuming checkpoint_dir is 'outputs' or similar root
        # We'll try to put it in a clean 'exported_models' dir parallel to checkpoints
        # If checkpoint_dir is outputs/pop909/checkpoints, we want outputs/pop909/exported_models?
        
        if "checkpoints" in ckpt_info.path.parts:
            # Find 'checkpoints' index
            idx = ckpt_info.path.parts.index("checkpoints")
            base_dir = Path(*ckpt_info.path.parts[:idx])
            export_dir = base_dir / "exported_models" / export_name
        else:
            # Fallback
            export_dir = self.checkpoint_dir / "exported_models" / export_name
            
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clean checkpoint (state_dict + hparams)
        # This mimics PL checkpoint structure but without optimizer states
        clean_checkpoint = {
            'state_dict': pl_module.state_dict(),
            'hyper_parameters': pl_module.hparams,
            'epoch': pl_module.current_epoch,
            'global_step': pl_module.global_step,
        }
        
        ckpt_path = export_dir / "model.ckpt"
        torch.save(clean_checkpoint, ckpt_path)
        
        # Save config.yaml if we can reconstruct it
        # Try to copy original config if found
        config_path = ckpt_info.path.parent / "config.yaml"
        if not config_path.exists():
             config_path = ckpt_info.path.parent.parent / "config.yaml"
        
        if config_path.exists():
            shutil.copy2(config_path, export_dir / "config.yaml")
        else:
            # Dump hparams as config
            with open(export_dir / "config.yaml", 'w') as f:
                # We need to convert config object to dict
                if hasattr(pl_module.config, '__dict__'):
                    # This is a rough approximation, might need better serialization
                    yaml.dump(pl_module.config, f)
        
        logger.info(f"Exported clean model to {export_dir}")
        
        # Rescan to find the new checkpoint
        self._scan_checkpoints()
        
        return str(export_dir)

    def get_current_model(self) -> Optional[PianoRollFractalGen]:
        """
        Get the currently loaded model.
        
        Returns:
            Current model or None if no model loaded
        """
        return self.current_model
    
    def get_current_checkpoint_info(self) -> Optional[CheckpointInfo]:
        """
        Get information about the currently loaded checkpoint.
        
        Returns:
            CheckpointInfo or None if no model loaded
        """
        return self.current_checkpoint
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_checkpoint = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model is None:
            return {"loaded": False}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.current_model.parameters())
        
        # Access config safely handling nested structure
        config = self.current_model.config
        
        # Check if config has architecture attribute (new style) or direct attributes (old style)
        if hasattr(config, 'architecture'):
            img_size_list = config.architecture.img_size_list
            embed_dim_list = config.architecture.embed_dim_list
            num_blocks_list = config.architecture.num_blocks_list
            num_heads_list = config.architecture.num_heads_list
        else:
            # Fallback or direct access
            img_size_list = getattr(config, 'img_size_list', None)
            embed_dim_list = getattr(config, 'embed_dim_list', None)
            num_blocks_list = getattr(config, 'num_blocks_list', None)
            num_heads_list = getattr(config, 'num_heads_list', None)
            
        # Generator settings usually in generator config
        if hasattr(config, 'generator'):
            generator_type_list = config.generator.generator_type_list
        else:
            generator_type_list = getattr(config, 'generator_type_list', None)
        
        info = {
            "loaded": True,
            "checkpoint": self.current_checkpoint.name if self.current_checkpoint else None,
            "step": self.current_checkpoint.step if self.current_checkpoint else None,
            "total_parameters": total_params,
            "parameters_millions": total_params / 1e6,
            "device": self.device,
            "config": {
                "img_size_list": img_size_list,
                "embed_dim_list": embed_dim_list,
                "num_blocks_list": num_blocks_list,
                "generator_type_list": generator_type_list,
            }
        }
        
        return info
