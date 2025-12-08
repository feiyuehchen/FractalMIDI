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
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.inference.inference import piano_roll_to_midi, create_generation_gif
from src.visualization.visualizer import piano_roll_to_image, create_growth_animation

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles generation requests and manages generation jobs.
    """
    
    def __init__(self, model_manager, config):
        """
        Initialize inference engine.
        """
        self.model_manager = model_manager
        self.config = config
        self.jobs: Dict[str, Dict] = {}  # job_id -> job_info
        self.output_dir = Path("src/web/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate(self, job_id: str, request):
        """
        Run generation job.
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
            # Check config structure
            default_iters = [8, 4, 2]
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'default_num_iter_list'):
                default_iters = self.config.model.default_num_iter_list
                
            num_iter_list = request.num_iter_list or default_iters
            
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
            import traceback
            traceback.print_exc()
            self.jobs[job_id].update({
                "status": "failed",
                "message": str(e),
                "failed_at": datetime.now().isoformat()
            })

    async def _generate_unconditional(self, model, request, job_id):
        """Generate unconditionally."""
        self.jobs[job_id]["message"] = "Generating from scratch..."
        self.jobs[job_id]["progress"] = 0.2
        
        loop = asyncio.get_event_loop()
        
        def progress_callback(step_data):
            level = step_data['level']
            step = step_data['step']
            p = 0.2 + 0.6 * ((level * 4 + step) / 14)
            self.jobs[job_id]["progress"] = min(0.9, p)

        def run_sample():
            with torch.no_grad():
                # Need bar_pos
                device = next(model.parameters()).device
                bar_pos = (torch.arange(request.length, device=device) % 16).unsqueeze(0)
                
                return model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or [8, 4, 2],
                    return_intermediates=request.create_gif,
                    callback=progress_callback,
                    bar_pos=bar_pos,
                    filter_threshold=request.filter_threshold
                )

        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            if isinstance(result, tuple):
                tensor_out, intermediates = result
            else:
                tensor_out = result
                intermediates = None
        else:
            tensor_out = result
            intermediates = None
        
        if isinstance(tensor_out, tuple): # Handle unexpected tuple return
             tensor_out = tensor_out[0]

        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        self.jobs[job_id]["progress"] = 0.8
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates
        }

    async def _generate_conditional(self, model, request, job_id):
        """Generate with conditioning."""
        # Just use inpainting logic but mask everything except condition region
        from src.web.backend.example_manager import ExampleManager
        
        if not request.condition_example_id:
            raise ValueError("condition_example_id required")
            
        example_manager = ExampleManager(self.config.examples.examples_dir)
        ex_pr = example_manager.load_example_piano_roll(
            request.condition_example_id,
            target_length=request.length
        )
        
        if ex_pr is None:
            raise ValueError(f"Could not load example: {request.condition_example_id}")
            
        # Adapt to (2, T, 128)
        if ex_pr.shape[0] >= 3:
            notes = ex_pr[:2]
        elif ex_pr.ndim == 2:
            notes = np.stack([ex_pr, ex_pr], axis=0) # Note=Vel
        else:
            notes = ex_pr
            
        c_start = max(0, request.condition_start or 0)
        c_end = min(request.length, request.condition_end or 64)
        
        # Mask: 1 = Generate, 0 = Keep
        mask_np = np.ones(request.length, dtype=np.float32)
        mask_np[c_start:c_end] = 0.0 
        
        device = next(model.parameters()).device
        initial_content = torch.from_numpy(notes).unsqueeze(0).to(device)
        inpaint_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        bar_pos = (torch.arange(request.length, device=device) % 16).unsqueeze(0)
        
        self.jobs[job_id]["message"] = "Conditional generation..."
        
        loop = asyncio.get_event_loop()
        def run_sample():
            with torch.no_grad():
                return model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or [8, 4, 2],
                    initial_content=initial_content,
                    inpaint_mask=inpaint_mask,
                    return_intermediates=request.create_gif,
                    bar_pos=bar_pos,
                    filter_threshold=request.filter_threshold
                )
        
        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            if isinstance(result, tuple):
                tensor_out, intermediates = result
            else:
                tensor_out = result
                intermediates = None
        else:
            tensor_out = result
            intermediates = None
            
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates
        }

    async def _generate_inpainting(self, model, request, job_id):
        """Generate with inpainting."""
        from src.web.backend.example_manager import ExampleManager
        
        if not request.inpaint_example_id:
            raise ValueError("inpaint_example_id required for inpainting")
        
        example_manager = ExampleManager(self.config.examples.examples_dir)
        piano_roll_np = example_manager.load_example_piano_roll(
            request.inpaint_example_id,
            target_length=request.length
        )
        
        if piano_roll_np is None:
            raise ValueError(f"Could not load example: {request.inpaint_example_id}")
            
        if piano_roll_np.ndim == 2:
            piano_roll_np = np.stack([piano_roll_np, piano_roll_np], axis=0)
        elif piano_roll_np.shape[0] >= 3:
            piano_roll_np = piano_roll_np[:2]
            
        # Mask
        mask_np = np.zeros((request.length,), dtype=np.float32)
        if request.inpaint_mask:
            for start, end in request.inpaint_mask:
                s = max(0, min(start, request.length))
                e = max(0, min(end, request.length))
                mask_np[s:e] = 1.0
        else:
            mask_np[request.length//2:] = 1.0
            
        device = next(model.parameters()).device
        initial_content = torch.from_numpy(piano_roll_np).unsqueeze(0).to(device)
        inpaint_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        bar_pos = (torch.arange(request.length, device=device) % 16).unsqueeze(0)
        
        self.jobs[job_id]["message"] = "Inpainting..."
        
        loop = asyncio.get_event_loop()
        def run_sample():
            with torch.no_grad():
                return model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or [8, 4, 2],
                    initial_content=initial_content,
                    inpaint_mask=inpaint_mask,
                    return_intermediates=request.create_gif,
                    bar_pos=bar_pos,
                    filter_threshold=request.filter_threshold
                )

        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            if isinstance(result, tuple):
                tensor_out, intermediates = result
            else:
                tensor_out = result
                intermediates = None
        else:
            tensor_out = result
            intermediates = None
            
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates,
            "original": piano_roll_np,
            "mask": mask_np
        }

    async def _generate_custom_inpainting(self, model, request, job_id):
        """Generate with custom user notes."""
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
                except: pass

        mask = np.ones_like(piano_roll, dtype=bool)
        if request.mask_points:
            for point in request.mask_points:
                if len(point) >= 2:
                    t, p = int(point[0]), int(point[1])
                    if 0 <= t < request.length and 0 <= p < 128:
                        mask[p, t] = False
                        
        notes = np.stack([piano_roll, piano_roll], axis=0)
        device = next(model.parameters()).device
        initial_content = torch.from_numpy(notes).unsqueeze(0).to(device)
        
        mask_1d = (~mask).any(axis=0).astype(np.float32) # 1 if any False
        
        inpaint_mask = torch.from_numpy(mask_1d).unsqueeze(0).to(device)
        bar_pos = (torch.arange(request.length, device=device) % 16).unsqueeze(0)
        
        self.jobs[job_id]["message"] = "Custom inpainting..."
        
        loop = asyncio.get_event_loop()
        def run_sample():
            with torch.no_grad():
                return model.sample(
                    batch_size=1,
                    length=request.length,
                    global_cond=None,
                    cfg=request.cfg,
                    temperature=request.temperature,
                    num_iter_list=request.num_iter_list or [8, 4, 2],
                    initial_content=initial_content,
                    inpaint_mask=inpaint_mask,
                    return_intermediates=request.create_gif,
                    bar_pos=bar_pos,
                    filter_threshold=request.filter_threshold
                )
        
        result = await loop.run_in_executor(None, run_sample)
        
        if request.create_gif:
            if isinstance(result, tuple):
                tensor_out, intermediates = result
            else:
                tensor_out = result
                intermediates = None
        else:
            tensor_out = result
            intermediates = None
            
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        return {
            "piano_roll": piano_roll_3ch,
            "intermediates": intermediates
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
            
            # Filter frames
            frames = []
            intermediates = result["intermediates"]
            for item in intermediates:
                if isinstance(item, dict) and not item.get('is_structure', False) and 'output' in item:
                    frame = item['output'][0]
                    T_f = frame.shape[1]
                    tempo_f = torch.ones(1, T_f, 128) * 0.5
                    full_frame = torch.cat([frame, tempo_f], dim=0)
                    frames.append(full_frame)
            
            if frames:
                create_growth_animation(
                    frames,
                    save_path=str(gif_path),
                    fps=15, # Fixed fps or from config
                    min_height=512,
                    show_progress=request.show_progress,
                    show_grid=request.show_grid,
                    pop_effect=True
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
        Run generation synchronously (for use in ThreadPoolExecutor).
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get model
        model = self.model_manager.get_current_model()
        if model is None:
            raise ValueError("No model loaded")
            
        device = next(model.parameters()).device
        
        # Define callback wrapper
        def callback(step_data):
            if streaming_frames is not None:
                streaming_frames.append(step_data)

        # Prepare common params
        num_iter_list = request.num_iter_list or [8, 4, 2]
        
        # Bar pos
        max_bar_len = 16
        if hasattr(self.config.model, 'architecture') and hasattr(self.config.model.architecture, 'max_bar_len'):
             max_bar_len = self.config.model.architecture.max_bar_len
        bar_pos = (torch.arange(request.length, device=device) % max_bar_len).unsqueeze(0)
        
        initial_content = None
        inpaint_mask = None
        
        if request.mode == 'conditional':
             from src.web.backend.example_manager import ExampleManager
             example_manager = ExampleManager(self.config.examples.examples_dir)
             ex_pr = example_manager.load_example_piano_roll(request.condition_example_id, target_length=request.length)
             
             if ex_pr.ndim == 2: notes = np.stack([ex_pr, ex_pr], axis=0)
             elif ex_pr.shape[0] >= 3: notes = ex_pr[:2]
             else: notes = ex_pr
             
             c_start = max(0, request.condition_start or 0)
             c_end = min(request.length, request.condition_end or 64)
             mask_np = np.ones(request.length, dtype=np.float32)
             mask_np[c_start:c_end] = 0.0
             
             initial_content = torch.from_numpy(notes).unsqueeze(0).to(device)
             inpaint_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
             
        elif request.mode == 'inpainting':
             from src.web.backend.example_manager import ExampleManager
             example_manager = ExampleManager(self.config.examples.examples_dir)
             piano_roll_np = example_manager.load_example_piano_roll(request.inpaint_example_id, target_length=request.length)
             
             if piano_roll_np is None:
                 raise ValueError(f"Could not load example: {request.inpaint_example_id}")

             if piano_roll_np.ndim == 2: piano_roll_np = np.stack([piano_roll_np, piano_roll_np], axis=0)
             elif piano_roll_np.shape[0] >= 3: piano_roll_np = piano_roll_np[:2]
             
             mask_np = np.zeros((request.length,), dtype=np.float32)
             if request.inpaint_mask:
                 for start, end in request.inpaint_mask:
                     s, e = max(0, min(start, request.length)), max(0, min(end, request.length))
                     mask_np[s:e] = 1.0
             else:
                 mask_np[request.length//2:] = 1.0
                 
             initial_content = torch.from_numpy(piano_roll_np).unsqueeze(0).to(device)
             inpaint_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.sample(
                batch_size=1,
                length=request.length,
                global_cond=None,
                cfg=request.cfg,
                temperature=request.temperature,
                num_iter_list=num_iter_list,
                initial_content=initial_content,
                inpaint_mask=inpaint_mask,
                return_intermediates=request.create_gif,
                callback=callback,
                bar_pos=bar_pos,
                filter_threshold=request.filter_threshold
            )
            
        # Post-processing
        if isinstance(result, tuple):
             tensor_out, intermediates = result
        else:
             tensor_out, intermediates = result, None
             
        piano_roll = tensor_out[0].cpu().numpy()
        T = piano_roll.shape[1]
        tempo_ch = np.ones((1, T, 128), dtype=np.float32) * 0.5
        piano_roll_3ch = np.concatenate([piano_roll, tempo_ch], axis=0)
        
        # Save output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        midi_path = job_dir / "output.mid"
        score = piano_roll_to_midi(piano_roll, velocity_threshold=0.1)
        score.dump_midi(str(midi_path))
        
        img_path = job_dir / "output.png"
        img = piano_roll_to_image(
            torch.from_numpy(piano_roll_3ch), 
            apply_colormap=True, 
            return_pil=True, 
            min_height=512,
            composite_tempo=True
        )
        img.save(str(img_path))
        
        return {
             'job_id': job_id,
             'midi_path': str(midi_path),
             'piano_roll': piano_roll_3ch,
             'intermediates': intermediates
        }
