import asyncio
import json
import logging
from typing import List, Dict
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import model related modules
from src.web.backend.config import AppConfig
from src.web.backend.model_manager import ModelManager
from src.web.backend.inference_api import InferenceEngine
from src.web.backend.td_bridge import td_bridge

logger = logging.getLogger(__name__)

class StreamableList(list):
    """List that triggers a callback on append."""
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()

    def append(self, item):
        super().append(item)
        if self.callback:
            # This is called from a thread, so we need to schedule the coroutine
            if asyncio.iscoroutinefunction(self.callback):
                asyncio.run_coroutine_threadsafe(self.callback(item), self.loop)
            else:
                self.loop.call_soon_threadsafe(self.callback, item)

class WebSocketStreamer:
    def __init__(self, websocket, model_manager: ModelManager, config: AppConfig):
        self.websocket = websocket
        self.model_manager = model_manager
        self.config = config
        
    async def stream_generation(self, request_data):
        """
        Handle generation request and stream updates via WebSocket.
        """
        try:
            # Parse request
            mode = request_data.get('mode', 'unconditional')
            length = request_data.get('length', 256)
            generator_type = request_data.get('generator_type', 'mar')
            
            # Get model
            model = self.model_manager.get_current_model()
            if not model:
                await self.websocket.send_json({
                    "status": "error",
                    "message": "No model loaded"
                })
                return

            # Setup generation parameters
            device = self.config.model.device
            num_iter_list = self.config.model.default_num_iter_list
            
            # Calculate total expected steps for progress bar
            h_patches = 128 // 16
            w_patches = max(1, length // 16)
            total_steps = h_patches * w_patches
            
            # Safety: if for some reason total_steps is too small
            if total_steps < 10: total_steps = 10
            
            cfg = request_data.get('cfg', 1.0)
            temperature = request_data.get('temperature', 1.0)
            
            await self.websocket.send_json({
                "status": "running",
                "progress": 0.05,
                "message": "Initializing generation..."
            })
            
            # Mutable state for progress
            state = {'current_step': 0}

            # Callback for streaming frames
            async def on_new_frame(frame_data):
                try:
                    # Increment step
                    state['current_step'] += 1
                    # Calculate progress
                    progress = min(0.95, 0.05 + (state['current_step'] / total_steps) * 0.9)
                    
                    output_tensor = frame_data['output']
                    if output_tensor.device.type == 'cuda':
                        output_tensor = output_tensor.cpu()
                        
                    is_structure = frame_data.get('is_structure', False)
                    
                    if is_structure:
                        # (B, 2, T) -> (2, T)
                        # [0]: Density, [1]: Tempo
                        if output_tensor.ndim == 3:
                            struct_data = output_tensor[0].numpy()
                            density = struct_data[0].tolist()
                            tempo = struct_data[1].tolist()
                            
                            await self.websocket.send_json({
                                "type": "structure_update",
                                "density": density,
                                "tempo": tempo,
                                "status": "generating",
                                "progress": progress,
                                "level": frame_data.get('level', 0)
                            })
                    else:
                        # (B, 2, T, 128) -> Note channel -> (T, 128)
                        # output_tensor[0, 0] is Note channel (T, 128)
                        if output_tensor.ndim == 4:
                            piano_roll = output_tensor[0, 0].numpy()
                            
                            notes = []
                            # shape (T, 128)
                            # rows=Time, cols=Pitch
                            rows, cols = np.where(piano_roll > 0.1)
                            
                            for t, p in zip(rows, cols):
                                notes.append({
                                    'pitch': int(p),
                                    'start': int(t),
                                    'duration': 1,
                                    'velocity': float(piano_roll[t, p])
                                })
                            
                            await self.websocket.send_json({
                                "type": "new_notes",
                                "notes": notes,
                                "status": "generating",
                                "progress": progress,
                                "message": f"Generating Level {frame_data.get('level', '?')}"
                            })
                            
                    # Stream Condition Influence if available
                    if frame_data.get('type') == 'condition_influence':
                        infl_data = frame_data['data']
                        cond_map = infl_data['cond_map'] # (B, T, C)
                        
                        # Calculate influence magnitude per time step: L2 norm over channels
                        # (B, T, C) -> (B, T)
                        influence_magnitude = torch.norm(cond_map.float(), dim=-1)
                        
                        # Normalize to 0-1 for visualization
                        if influence_magnitude.max() > 0:
                            influence_magnitude = influence_magnitude / influence_magnitude.max()
                            
                        influence_list = influence_magnitude[0].tolist()
                        
                        await self.websocket.send_json({
                            "type": "condition_influence",
                            "level": infl_data['level'],
                            "influence": influence_list
                        })

                except Exception as e:
                    logger.error(f"Frame streaming error: {e}")

            # Initialize inference engine
            engine = InferenceEngine(self.model_manager, self.config)
            
            # Create a request object compatible with engine
            from src.web.backend.app import GenerationRequest
            req = GenerationRequest(**request_data)
            
            # Create special container
            streaming_frames = StreamableList(on_new_frame) if req.create_gif else None
            
            # Run generation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: engine.generate_synchronous(req, streaming_frames=streaming_frames)
            )
            
            # Load the generated MIDI to get notes
            import symusic
            # Get path from result dict
            midi_path_str = result.get('midi_path')
            if not midi_path_str:
                 raise ValueError(f"No midi_path in result: {result}")

            score = symusic.Score(midi_path_str)
            notes_data = []
            for track in score.tracks:
                for note in track.notes:
                    notes_data.append({
                        'pitch': note.pitch,
                        'start': note.time / score.ticks_per_quarter * 4, # approx 16th steps
                        'duration': note.duration / score.ticks_per_quarter * 4,
                        'velocity': note.velocity / 127.0
                    })
            
            # Sort by start time
            notes_data.sort(key=lambda x: x['start'])
            
            # Broadcast to TouchDesigner
            await td_bridge.broadcast_notes(notes_data)
            
            # Send final result notes
            await self.websocket.send_json({
                "type": "new_notes",
                "notes": notes_data
            })
            
            # Send final completion
            await self.websocket.send_json({
                "status": "completed",
                "progress": 1.0,
                "midi_url": f"/outputs/{result['job_id']}/output.mid",
                "image_url": f"/outputs/{result['job_id']}/output.png",
                "gif_url": f"/outputs/{result['job_id']}/output.gif" if req.create_gif else None
            })
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            await self.websocket.send_json({
                "status": "error",
                "message": str(e)
            })
