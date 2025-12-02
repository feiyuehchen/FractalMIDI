"""
Inference engine for FractalMIDI generation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, AsyncGenerator
import logging
from datetime import datetime
import asyncio

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import piano_roll_to_midi, create_generation_gif
from visualizer import piano_roll_to_image, create_growth_animation
from models import conditional_generation, inpainting_generation

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles generation requests and manages generation jobs.
    """
    
    def __init__(self, model_manager, config):
        """
        Initialize inference engine.
        
        Args:
            model_manager: ModelManager instance
            config: AppConfig instance
        """
        self.model_manager = model_manager
        self.config = config
        self.jobs: Dict[str, Dict] = {}  # job_id -> job_info
        self.output_dir = Path("web/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate(self, job_id: str, request):
        """
        Run generation job.
        
        Args:
            job_id: Unique job ID
            request: GenerationRequest object
        """
        # Initialize job
        self.jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Starting generation...",
            "started_at": datetime.now().isoformat()
        }
        
        try:
            # Get model
            model = self.model_manager.get_current_model()
            if model is None:
                raise ValueError("No model loaded")
            
            # Prepare generation parameters
            num_iter_list = request.num_iter_list or self.config.model.default_num_iter_list
            
            # Update status
            self.jobs[job_id]["message"] = "Preparing generation..."
            self.jobs[job_id]["progress"] = 0.1
            
            # Generate based on mode
            if request.mode == "unconditional":
                result = await self._generate_unconditional(model, request, job_id)
            elif request.mode == "conditional":
                result = await self._generate_conditional(model, request, job_id)
            elif request.mode == "inpainting":
                result = await self._generate_inpainting(model, request, job_id)
            elif request.mode == "inpainting_custom":
                result = await self._generate_custom_inpainting(model, request, job_id)
            else:
                raise ValueError(f"Unknown mode: {request.mode}")
            
            # Save outputs
            self.jobs[job_id]["message"] = "Saving outputs..."
            self.jobs[job_id]["progress"] = 0.9
            
            output_paths = await self._save_outputs(job_id, result, request)
            
            # Update job status
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "message": "Generation completed",
                "completed_at": datetime.now().isoformat(),
                **output_paths
            })
            
        except Exception as e:
            logger.error(f"Generation error for job {job_id}: {e}")
            self.jobs[job_id].update({
                "status": "failed",
                "message": str(e),
                "failed_at": datetime.now().isoformat()
            })

    async def _generate_custom_inpainting(self, model, request, job_id):
        """Generate with custom user notes and mask."""
        
        # 1. Construct Piano Roll from user_notes
        piano_roll = np.zeros((128, request.length), dtype=np.float32)
        if request.user_notes:
            for n in request.user_notes:
                try:
                    p = int(n['pitch'])
                    s = int(n['start'])
                    d = int(n.get('duration', 1))
                    v = float(n['velocity'])
                    if 0 <= p < 128 and s < request.length:
                        end = min(s + d, request.length)
                        piano_roll[p, s:end] = v
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid note: {n} ({e})")

        # 2. Construct Mask from mask_points
        # Mask: 1 = Keep, 0 = Regenerate
        mask = np.ones_like(piano_roll, dtype=bool)
        if request.mask_points:
            for point in request.mask_points:
                if len(point) >= 2:
                    t, p = int(point[0]), int(point[1])
                    if 0 <= t < request.length and 0 <= p < 128:
                        mask[p, t] = False
                        
        self.jobs[job_id]["message"] = "Inpainting custom region..."
        self.jobs[job_id]["progress"] = 0.2
        
        # Run inpainting
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: inpainting_generation(
                model=model,
                piano_roll=torch.from_numpy(piano_roll).unsqueeze(0).unsqueeze(0),
                mask=torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
                num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                cfg=request.cfg,
                temperature=request.temperature,
                return_intermediates=request.create_gif
            )
        )
        
        if request.create_gif:
            piano_roll_result, intermediates = result
        else:
            piano_roll_result = result
            intermediates = None
        
        self.jobs[job_id]["progress"] = 0.8
        
        return {
            "piano_roll": piano_roll_result[0, 0].cpu().numpy(),
            "intermediates": intermediates
        }

    
    async def _generate_unconditional(self, model, request, job_id):
        """Generate unconditionally."""
        self.jobs[job_id]["message"] = "Generating from scratch..."
        self.jobs[job_id]["progress"] = 0.2
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def run_sample():
            with torch.no_grad():
                return model.model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                    return_intermediates=request.create_gif
                )

        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            # result is (final_output, intermediates)
            # final_output: (B, 2, T, 128)
            tensor_out, intermediates = result
        else:
            tensor_out = result
            intermediates = None
        
        # Convert (B, 2, T, 128) -> (2, T, 128) -> (3, T, 128) with dummy tempo
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        self.jobs[job_id]["progress"] = 0.8
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates
        }

    async def _generate_inpainting(self, model, request, job_id):
        """Generate with inpainting."""
        from web.backend.example_manager import ExampleManager
        
        if not request.inpaint_example_id:
            raise ValueError("inpaint_example_id required for inpainting")
        
        # Load example
        example_manager = ExampleManager(self.config.examples.examples_dir)
        piano_roll_np = example_manager.load_example_piano_roll(
            request.inpaint_example_id,
            target_length=request.length
        ) # Expected (3, T, 128) or (2, T, 128) or (128, T)
        
        if piano_roll_np is None:
            raise ValueError(f"Could not load example: {request.inpaint_example_id}")
            
        # Adapt shape if needed
        if piano_roll_np.ndim == 2 and piano_roll_np.shape[0] == 128: # (128, T)
            # Convert to (2, T, 128)
            # Assume Velocity = Note
            T = piano_roll_np.shape[1]
            note = piano_roll_np.T
            vel = piano_roll_np.T
            piano_roll_np = np.stack([note, vel], axis=0) # (2, T, 128)
        elif piano_roll_np.ndim == 3 and piano_roll_np.shape[0] == 3:
            # Drop tempo for input
            piano_roll_np = piano_roll_np[:2]
            
        # Create mask
        # mask: 1.0 where we want to generate (inpainting mask)
        mask_np = np.zeros((request.length,), dtype=np.float32)
        if request.inpaint_mask:
            for start, end in request.inpaint_mask:
                s = max(0, min(start, request.length))
                e = max(0, min(end, request.length))
                mask_np[s:e] = 1.0
        else:
            # Default mask: last 50%?
            mask_np[request.length//2:] = 1.0
            
        # Convert to tensors
        device = next(model.parameters()).device
        initial_content = torch.from_numpy(piano_roll_np).unsqueeze(0).to(device) # (1, 2, T, 128)
        inpaint_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device) # (1, T)

        self.jobs[job_id]["message"] = "Inpainting masked regions..."
        self.jobs[job_id]["progress"] = 0.2
        
        loop = asyncio.get_event_loop()
        def run_sample():
            with torch.no_grad():
                return model.model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                    initial_content=initial_content,
                    inpaint_mask=inpaint_mask,
                    return_intermediates=request.create_gif
                )

        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            tensor_out, intermediates = result
        else:
            tensor_out = result
            intermediates = None
            
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        self.jobs[job_id]["progress"] = 0.8
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates,
            "original": piano_roll_np,
            "mask": mask_np
        }
    
    async def _save_outputs(self, job_id: str, result: Dict, request) -> Dict:
        """Save generation outputs."""
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        piano_roll = result["piano_roll"]
        
        # Save MIDI
        midi_path = job_dir / "output.mid"
        score = piano_roll_to_midi(piano_roll, velocity_threshold=0.1)
        score.dump_midi(str(midi_path))
        
        # Save image
        img_path = job_dir / "output.png"
        img = piano_roll_to_image(
            torch.from_numpy(piano_roll),
            apply_colormap=True,
            return_pil=True,
            min_height=512
        )
        img.save(str(img_path))
        
        # Save GIF if requested
        gif_path = None
        if request.create_gif and result.get("intermediates"):
            gif_path = job_dir / "output.gif"
            create_growth_animation(
                result["intermediates"],
                save_path=str(gif_path),
                fps=self.config.generation.gif_fps,
                min_height=512,
                show_progress=request.show_progress,
                show_grid=request.show_grid,
                quality=self.config.generation.gif_quality
            )
        
        return {
            "midi_url": f"/outputs/{job_id}/output.mid",
            "image_url": f"/outputs/{job_id}/output.png",
            "gif_url": f"/outputs/{job_id}/output.gif" if gif_path else None
        }
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status."""
        return self.jobs.get(job_id)
    
    def generate_synchronous(self, request, streaming_frames=None):
        """
        Synchronous version of generation.
        
        Args:
            streaming_frames: Optional list-like object to append intermediate frames to.
        """
        job_id = f"sync_{datetime.now().timestamp()}"
        
        model = self.model_manager.get_current_model()
        if model is None:
            raise ValueError("No model loaded")
        
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare intermediates container if streaming
        _intermediates_list = None
        if streaming_frames is not None:
            _intermediates_list = {
                'frames': streaming_frames,
                'canvas': torch.full((1, 1, 128, request.length), fill_value=-1.0).to(next(model.parameters()).device)
            }

        result = None

        # Run sampling
        if request.mode == "unconditional":
             result = model.sample(
                batch_size=1,
                cond_list=None,
                num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                cfg=request.cfg,
                temperature=request.temperature,
                filter_threshold=self.config.model.default_filter_threshold,
                target_width=request.length,
                return_intermediates=request.create_gif,
                _intermediates_list=_intermediates_list
            )
        elif request.mode == "inpainting_custom":
            # Custom Inpainting Sync Logic
            piano_roll = np.zeros((128, request.length), dtype=np.float32)
            if request.user_notes:
                for n in request.user_notes:
                    try:
                        p, s = int(n['pitch']), int(n['start'])
                        d, v = int(n.get('duration', 1)), float(n['velocity'])
                        if 0 <= p < 128 and s < request.length:
                            piano_roll[p, s:min(s+d, request.length)] = v
                    except: pass
            
            mask = np.ones_like(piano_roll, dtype=bool)
            if request.mask_points:
                for point in request.mask_points:
                    if len(point) >= 2:
                        t, p = int(point[0]), int(point[1])
                        if 0 <= t < request.length and 0 <= p < 128:
                            mask[p, t] = False
            
            device = next(model.parameters()).device
            piano_roll_t = torch.from_numpy(piano_roll).unsqueeze(0).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
            
            result = inpainting_generation(
                model=model,
                piano_roll=piano_roll_t,
                mask=mask_t,
                num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                cfg=request.cfg,
                temperature=request.temperature,
                return_intermediates=request.create_gif,
                _intermediates_list=_intermediates_list
            )
        elif request.mode == "conditional":
             # Conditional Sync Logic - Treated as "Keep Region" Inpainting
             from web.backend.example_manager import ExampleManager
             
             if not request.condition_example_id:
                 raise ValueError("condition_example_id required")
             
             example_manager = ExampleManager(self.config.examples.examples_dir)
             
             # Load example up to requested length
             ex_pr = example_manager.load_example_piano_roll(
                 request.condition_example_id,
                 target_length=request.length
             )
             
             if ex_pr is None:
                 raise ValueError(f"Could not load example: {request.condition_example_id}")
             
             # Initialize Canvas (Empty) and Mask (Full Mask = Generate All)
             piano_roll = np.zeros((128, request.length), dtype=np.float32)
             mask = np.ones_like(piano_roll, dtype=bool) # True = Masked (Generate)
             
             # Determine region to KEEP
             c_start = request.condition_start if request.condition_start is not None else 0
             c_end = request.condition_end if request.condition_end is not None else (request.condition_length or 64)
             
             # Clamp bounds
             c_start = max(0, min(c_start, request.length))
             c_end = max(c_start, min(c_end, request.length))
             
             # Copy example content to canvas and unmask
             ex_len = ex_pr.shape[1]
             src_start = c_start
             src_end = min(c_end, ex_len)
             
             if src_end > src_start:
                 piano_roll[:, src_start:src_end] = ex_pr[:, src_start:src_end]
                 mask[:, src_start:src_end] = False # False = Keep (Unmasked)
             
             device = next(model.parameters()).device
             piano_roll_t = torch.from_numpy(piano_roll).unsqueeze(0).unsqueeze(0).to(device)
             mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
             
             # Use inpainting_generation (it handles masking)
             result = inpainting_generation(
                model=model,
                piano_roll=piano_roll_t,
                mask=mask_t,
                num_iter_list=request.num_iter_list or self.config.model.default_num_iter_list,
                cfg=request.cfg,
                temperature=request.temperature,
                return_intermediates=request.create_gif,
                _intermediates_list=_intermediates_list
            )
        else:
             raise NotImplementedError(f"Sync mode not impl: {request.mode}")
        
        logger.info(f"Sample result type: {type(result)}")
        
        if request.create_gif:
            if isinstance(result, tuple):
                piano_roll, intermediates = result
            else:
                piano_roll = result
                intermediates = None
        else:
            piano_roll = result
            intermediates = None
            
        # Handle different tensor dimensions robustly
        if hasattr(piano_roll, 'shape'):
            if piano_roll.ndim == 4: # (B, C, H, W)
                piano_roll = piano_roll[0, 0]
            elif piano_roll.ndim == 3: # (B, H, W) or (C, H, W)
                piano_roll = piano_roll[0]
            # If ndim == 2 (H, W), keep as is
            
        piano_roll = piano_roll.cpu().numpy()
        
        # Save MIDI
        midi_path = job_dir / "output.mid"
        score = piano_roll_to_midi(piano_roll, velocity_threshold=0.1)
        score.dump_midi(str(midi_path))
        
        # Save Image
        img_path = job_dir / "output.png"
        img = piano_roll_to_image(
            torch.from_numpy(piano_roll),
            apply_colormap=True,
            return_pil=True,
            min_height=512
        )
        img.save(img_path)
        
        # Save GIF
        gif_path = None
        if request.create_gif and intermediates:
            gif_path = job_dir / "generation.gif"
            
            frames = []
            for i, step_pr in enumerate(intermediates):
                if isinstance(step_pr, dict):
                    pr_tensor = step_pr['output']
                else:
                    pr_tensor = step_pr
                    
                step_img = piano_roll_to_image(
                    pr_tensor[0, 0].cpu(),
                    apply_colormap=True,
                    return_pil=True,
                    min_height=512
                )
                frames.append(step_img)
            
            if frames:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0
                )

        return {
            "job_id": job_id,
            "midi_path": str(midi_path)
        }
