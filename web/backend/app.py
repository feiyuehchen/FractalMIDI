"""
FastAPI application for FractalMIDI web interface.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from pathlib import Path
import logging
import asyncio
import uuid
from datetime import datetime
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.backend.config import get_config
from web.backend.model_manager import ModelManager, CheckpointInfo
from web.backend.example_manager import ExampleManager
from web.backend.inference_api import InferenceEngine
from web.backend.streaming_utils import WebSocketStreamer
from web.backend.td_bridge import td_bridge

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="FractalMIDI API",
    description="API for FractalMIDI hierarchical music generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allow_origins,
    allow_credentials=config.cors.allow_credentials,
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)

# Initialize managers
model_manager = ModelManager(
    checkpoint_dir=config.model.checkpoint_dir,
    device=config.model.device
)

example_manager = ExampleManager(
    examples_dir=config.examples.examples_dir,
    max_examples=config.examples.max_examples
)

inference_engine = InferenceEngine(
    model_manager=model_manager,
    config=config
)

# Mount static files
if config.static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(config.static_dir)), name="static")
else:
    logger.warning(f"Static directory not found: {config.static_dir}")

# Mount outputs directory
from fastapi.staticfiles import StaticFiles as SF
output_dir = Path("web/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", SF(directory=str(output_dir)), name="outputs")


# ==============================================================================
# Request/Response Models
# ==============================================================================

class LoadModelRequest(BaseModel):
    """Request for loading a model."""
    checkpoint_name: Optional[str] = None


class GenerationRequest(BaseModel):
    """Request for generation."""
    mode: str = Field(..., description="Generation mode: unconditional, conditional, or inpainting")
    generator_type: str = Field("mar", description="Generator type: mar or ar")
    scan_order: str = Field("row_major", description="Scan order for AR: row_major or column_major")
    length: int = Field(256, description="Generation length in time steps")
    temperature: float = Field(1.0, description="Sampling temperature")
    cfg: float = Field(1.0, description="Classifier-free guidance scale")
    num_iter_list: Optional[List[int]] = Field(None, description="Number of iterations per level")
    
    # Conditional generation
    condition_example_id: Optional[str] = Field(None, description="Example ID for conditioning")
    condition_length: Optional[int] = Field(32, description="Length of condition in time steps")
    condition_start: Optional[int] = Field(0, description="Start time of condition region")
    condition_end: Optional[int] = Field(64, description="End time of condition region")
    
    # Inpainting
    inpaint_example_id: Optional[str] = Field(None, description="Example ID for inpainting")
    inpaint_mask: Optional[List[List[int]]] = Field(None, description="Inpainting mask regions [[start, end], ...]")
    
    # Custom Editing (Canvas)
    user_notes: Optional[List[Dict]] = Field(None, description="User provided notes for inpainting [{pitch, start, duration, velocity}, ...]")
    mask_points: Optional[List[List[int]]] = Field(None, description="List of [time, pitch] points to mask for pixel-level inpainting")
    
    # Visualization
    create_gif: bool = Field(True, description="Whether to create GIF animation")
    show_progress: bool = Field(True, description="Show progress indicator in GIF")
    show_grid: bool = Field(False, description="Show grid in GIF")


class GenerationResponse(BaseModel):
    """Response for generation."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    message: Optional[str] = None
    midi_url: Optional[str] = None
    image_url: Optional[str] = None
    gif_url: Optional[str] = None
    progress: float = 0.0


class ModelInfo(BaseModel):
    """Model information."""
    loaded: bool
    checkpoint: Optional[str] = None
    step: Optional[int] = None
    total_parameters: Optional[int] = None
    parameters_millions: Optional[float] = None
    device: Optional[str] = None
    config: Optional[Dict] = None


class ExampleResponse(BaseModel):
    """Example information response."""
    id: str
    name: str
    duration_seconds: float
    num_notes: int
    time_steps: int
    pitch_range: tuple
    thumbnail_url: Optional[str] = None
    tags: Optional[List[str]] = None


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint - Redirect to UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running",
        "model_loaded": model_manager.get_current_model() is not None,
        "num_checkpoints": len(model_manager.list_checkpoints()),
        "num_examples": len(example_manager.list_examples()),
        "device": config.model.device,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/models/list")
async def list_models():
    """List available model checkpoints."""
    checkpoints = model_manager.list_checkpoints()
    return {
        "checkpoints": [
            {
                "name": ckpt.name,
                "step": ckpt.step,
                "epoch": ckpt.epoch,
                "file_size_mb": ckpt.file_size_mb,
                "generator_types": ckpt.generator_types,
                "scan_order": ckpt.scan_order
            }
            for ckpt in checkpoints
        ]
    }


@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    """Load a model checkpoint."""
    try:
        checkpoint_name = request.checkpoint_name
        model = model_manager.load_checkpoint(checkpoint_name)
        info = model_manager.get_model_info()
        return {
            "status": "success",
            "message": f"Loaded checkpoint: {info['checkpoint']}",
            "model_info": info
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    info = model_manager.get_model_info()
    return ModelInfo(**info)


@app.get("/api/examples/list")
async def list_examples(limit: Optional[int] = None):
    """List available MIDI examples."""
    examples = example_manager.list_examples(limit=limit)
    return {
        "examples": [
            ExampleResponse(
                id=ex.id,
                name=ex.name,
                duration_seconds=ex.duration_seconds,
                num_notes=ex.num_notes,
                time_steps=ex.time_steps,
                pitch_range=ex.pitch_range,
                thumbnail_url=f"/static/examples/{ex.thumbnail_path}" if ex.thumbnail_path else None,
                tags=ex.tags
            )
            for ex in examples
        ]
    }


@app.get("/api/examples/{example_id}")
async def get_example(example_id: str):
    """Get information about a specific example."""
    example = example_manager.get_example(example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")
    
    return ExampleResponse(
        id=example.id,
        name=example.name,
        duration_seconds=example.duration_seconds,
        num_notes=example.num_notes,
        time_steps=example.time_steps,
        pitch_range=example.pitch_range,
        thumbnail_url=f"/static/examples/{example.thumbnail_path}" if example.thumbnail_path else None,
        tags=example.tags
    )

@app.get("/api/examples/{example_id}/notes")
async def get_example_notes(example_id: str):
    """Get notes for a specific example for visualization."""
    import symusic
    
    example = example_manager.get_example(example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")
    
    try:
        midi_path = example_manager.examples_dir / example.file_path
        score = symusic.Score(str(midi_path))
        
        ticks_per_quarter = score.ticks_per_quarter
        ticks_per_16th = ticks_per_quarter // 4
        
        notes_list = []
        for track in score.tracks:
            if track.is_drum: continue
            for note in track.notes:
                notes_list.append({
                    "pitch": note.pitch,
                    "start": int(note.time // ticks_per_16th),
                    "duration": int(note.duration // ticks_per_16th),
                    "velocity": note.velocity / 127.0
                })
        
        return {"notes": notes_list}
    except Exception as e:
        logger.error(f"Error getting notes for example {example_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting notes: {str(e)}")


@app.post("/api/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Start a generation job.
    
    This endpoint starts a background generation job and returns immediately
    with a job ID. Use the /api/generate/{job_id} endpoint to check status.
    """
    try:
        # Create job
        job_id = str(uuid.uuid4())
        
        # Start generation in background
        background_tasks.add_task(
            inference_engine.generate,
            job_id=job_id,
            request=request
        )
        
        return GenerationResponse(
            job_id=job_id,
            status="pending",
            message="Generation job started"
        )
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate/{job_id}", response_model=GenerationResponse)
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    status = inference_engine.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time generation with progress updates.
    """
    await websocket.accept()
    
    streamer = WebSocketStreamer(websocket, model_manager, config)
    
    try:
        # Receive generation request
        data = await websocket.receive_json()
        
        # Use the new streaming helper
        await streamer.stream_generation(data)
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/touchdesigner")
async def websocket_touchdesigner(websocket: WebSocket):
    """
    WebSocket endpoint for TouchDesigner integration.
    Receives generated notes in real-time.
    """
    await websocket.accept()
    await td_bridge.connect(websocket)
    
    try:
        while True:
            # Keep connection alive, maybe handle incoming control messages later
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await td_bridge.disconnect(websocket)


# ==============================================================================
# Startup/Shutdown Events
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting FractalMIDI API server")
    
    # Don't auto-load model to prevent blocking startup
    # User can select model from UI
    logger.info("Server started ready (waiting for model selection)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FractalMIDI API server")
    model_manager.unload_model()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=config.server.workers,
        log_level=config.server.log_level
    )
