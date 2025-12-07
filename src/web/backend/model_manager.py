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
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import ALL classes that might be in the YAML to ensure UnsafeLoader can find them
from src.training.trainer import FractalMIDILightningModule, FractalTrainerConfig, OptimizerConfig, SchedulerConfig
from src.models.model_config import FractalModelConfig, ArchitectureConfig, GeneratorConfig, PianoRollConfig, TrainingConfig
# Alias for backward compatibility if checkpoints refer to old names
ModelConfig = FractalModelConfig 
from src.models.temporal_fractal import TemporalFractalNetwork as PianoRollFractalGen

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
        self.current_config: Optional[FractalModelConfig] = None
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
                relative_path = ckpt_path.relative_to(self.checkpoint_dir)
                parts = relative_path.parts
                
                name_parts = [p for p in parts[:-1] if p != "checkpoints"]
                
                name_prefix = "/".join(name_parts)
                
                filename = ckpt_path.stem 
                
                short_name = filename
                if "step" in filename:
                    step_str = filename.split("-")[0] 
                    short_name = step_str
                
                display_name = f"{name_prefix}/{short_name}" if name_prefix else short_name
                
                name = display_name 
                step = None
                epoch = None
                
                if "step" in filename:
                    try:
                        if "step=" in filename:
                            step = int(filename.split("step=")[1].split("-")[0].split(".")[0])
                        elif "step_" in filename:
                            step = int(filename.split("step_")[1].split("-")[0].split(".")[0])
                    except:
                        pass
                
                if "epoch" in filename:
                    try:
                        if "epoch=" in filename:
                            epoch = int(filename.split("epoch=")[1].split("-")[0].split(".")[0])
                        elif "epoch_" in filename:
                            epoch = int(filename.split("epoch_")[1].split("-")[0].split(".")[0])
                    except:
                        pass
                
                file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                
                generator_types = None
                scan_order = None
                
                config_locations = [
                    ckpt_path.parent / "config.yaml", 
                    ckpt_path.parent.parent / "config.yaml",  
                ]
                
                if "checkpoints" in parts:
                    ckpt_idx = parts.index("checkpoints")
                    if len(parts) > ckpt_idx + 1: 
                        version_dir = parts[ckpt_idx + 1]
                        base_output_dir = Path(*parts[:ckpt_idx])
                        log_config_path = self.checkpoint_dir / base_output_dir / "logs" / version_dir / "hparams.yaml"
                        config_locations.append(log_config_path)
                
                config_data = None
                for conf_path in config_locations:
                    if conf_path.exists():
                        try:
                            with open(conf_path, 'r') as f:
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
                    model_conf = None
                    
                    def get_val(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key)
                        return getattr(obj, key, None)

                    if isinstance(config_data, dict) and 'model' in config_data:
                         model_conf = config_data['model']
                    elif hasattr(config_data, 'model'): 
                         model_conf = config_data.model
                    else:
                         conf_obj = get_val(config_data, 'config')
                         if conf_obj:
                             model_conf = get_val(conf_obj, 'model')

                    if model_conf:
                        # Handle both flat and nested config structures
                        # New structure: model.generator.generator_type_list
                        # Old structure: model.generator_type_list
                        
                        gen_conf = get_val(model_conf, 'generator')
                        if gen_conf:
                            gen_list = get_val(gen_conf, 'generator_type_list')
                            scan_o = get_val(gen_conf, 'scan_order')
                        else:
                            gen_list = get_val(model_conf, 'generator_type_list')
                            scan_o = get_val(model_conf, 'scan_order')

                        if gen_list:
                            if isinstance(gen_list, (list, tuple)):
                                generator_types = list(gen_list)
                            
                        if scan_o:
                            scan_order = scan_o
                    
                    if "version_9" in name and generator_types is None:
                        generator_types = ["mar", "mar", "mar", "mar"]
                
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
        """
        return list(self.available_checkpoints.values())
    
    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> PianoRollFractalGen:
        """
        Load a checkpoint by name.
        """
        if checkpoint_name is None:
            if not self.available_checkpoints:
                raise ValueError("No checkpoints available")
            
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
        
        if self.current_checkpoint and self.current_checkpoint.path == checkpoint_info.path:
            if self.current_model is not None:
                logger.info(f"Checkpoint already loaded: {checkpoint_info.name}")
                return self.current_model
        
        logger.info(f"Loading checkpoint: {checkpoint_info.path}")
        
        try:
            pl_module = FractalMIDILightningModule.load_from_checkpoint(
                str(checkpoint_info.path),
                map_location=self.device,
                strict=False
            )
            
            model = pl_module.model
            model.eval()
            model.to(self.device)
            
            self.current_model = model
            self.current_config = pl_module.config.model if hasattr(pl_module, 'config') else None
            self.current_checkpoint = checkpoint_info
            
            logger.info(f"Successfully loaded checkpoint: {checkpoint_info.name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def export_model(self, checkpoint_name: str, export_name: str):
        """
        Export a clean model checkpoint.
        """
        if checkpoint_name not in self.available_checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")
        
        ckpt_info = self.available_checkpoints[checkpoint_name]
        
        pl_module = FractalMIDILightningModule.load_from_checkpoint(
            str(ckpt_info.path),
            map_location="cpu", 
            strict=False
        )
        
        if "checkpoints" in ckpt_info.path.parts:
            idx = ckpt_info.path.parts.index("checkpoints")
            base_dir = Path(*ckpt_info.path.parts[:idx])
            export_dir = base_dir / "exported_models" / export_name
        else:
            export_dir = self.checkpoint_dir / "exported_models" / export_name
            
        export_dir.mkdir(parents=True, exist_ok=True)
        
        clean_checkpoint = {
            'state_dict': pl_module.state_dict(),
            'hyper_parameters': pl_module.hparams,
            'epoch': pl_module.current_epoch,
            'global_step': pl_module.global_step,
        }
        
        ckpt_path = export_dir / "model.ckpt"
        torch.save(clean_checkpoint, ckpt_path)
        
        config_path = ckpt_info.path.parent / "config.yaml"
        if not config_path.exists():
             config_path = ckpt_info.path.parent.parent / "config.yaml"
        
        if config_path.exists():
            shutil.copy2(config_path, export_dir / "config.yaml")
        else:
            with open(export_dir / "config.yaml", 'w') as f:
                if hasattr(pl_module.config, '__dict__'):
                    yaml.dump(pl_module.config, f)
        
        logger.info(f"Exported clean model to {export_dir}")
        
        self._scan_checkpoints()
        
        return str(export_dir)

    def get_current_model(self) -> Optional[PianoRollFractalGen]:
        return self.current_model
    
    def get_current_checkpoint_info(self) -> Optional[CheckpointInfo]:
        return self.current_checkpoint
    
    def unload_model(self):
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_config = None
            self.current_checkpoint = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
    
    def get_model_info(self) -> Dict:
        if self.current_model is None:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.current_model.parameters())
        
        config = self.current_config
        
        img_size_list = None
        embed_dim_list = None
        num_blocks_list = None
        generator_type_list = None
        
        if config:
            if hasattr(config, 'architecture'):
                img_size_list = config.architecture.img_size_list
                embed_dim_list = config.architecture.embed_dim_list
                num_blocks_list = config.architecture.num_blocks_list
            else:
                img_size_list = getattr(config, 'img_size_list', None)
                embed_dim_list = getattr(config, 'embed_dim_list', None)
                num_blocks_list = getattr(config, 'num_blocks_list', None)
                
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
