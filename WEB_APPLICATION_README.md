# FractalMIDI Web Application

## Complete Implementation Guide

This document provides a comprehensive guide to the FractalMIDI web application, including setup, usage, and deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [API Documentation](#api-documentation)
6. [Frontend Features](#frontend-features)
7. [Docker Deployment](#docker-deployment)
8. [TouchDesigner Integration](#touchdesigner-integration)
9. [Development](#development)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The FractalMIDI web application provides an interactive interface for hierarchical music generation. It features:

- **Real-time generation** with WebSocket streaming
- **Three generation modes**: Unconditional, Conditional, Inpainting
- **Interactive piano roll** visualization
- **High-quality GIF** animations of the generation process
- **Model selection** and parameter control
- **Validation examples** for conditional/inpainting
- **TouchDesigner integration** for interactive art installations

---

## Architecture

```
FractalMIDI/
├── web/
│   ├── backend/
│   │   ├── app.py                 # FastAPI main application
│   │   ├── config.py              # Configuration management
│   │   ├── model_manager.py       # Model loading and management
│   │   ├── example_manager.py     # Validation examples
│   │   └── inference_api.py       # Generation engine
│   ├── frontend/
│   │   └── static/
│   │       ├── index.html         # Main interactive page
│   │       ├── about.html         # About page
│   │       ├── docs.html          # Documentation
│   │       ├── css/style.css      # Styling
│   │       └── js/
│   │           ├── main.js        # Main application logic
│   │           ├── pianoroll.js   # Canvas renderer
│   │           └── ...            # Other utilities
│   └── requirements_web.txt       # Web dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
└── nginx.conf                     # Nginx reverse proxy
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Trained FractalMIDI model checkpoints

### Step 1: Install Dependencies

```bash
# Install main FractalMIDI dependencies (if not already installed)
pip install -r requirements.txt

# Install web application dependencies
pip install -r web/requirements_web.txt
```

### Step 2: Prepare Checkpoints

Place your trained model checkpoints in `outputs/checkpoints/`:

```bash
mkdir -p outputs/checkpoints
# Copy your .ckpt files here
```

### Step 3: Prepare Validation Examples (Optional)

For conditional and inpainting modes, prepare MIDI examples:

```bash
mkdir -p dataset/validation_examples
# Copy MIDI files here
```

The example manager will automatically scan and create metadata.

---

## Running the Application

### Development Mode

```bash
cd web/backend
python app.py
```

The server will start on `http://localhost:8000`.

### Production Mode with Uvicorn

```bash
cd web/backend
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Access the Application

Open your browser and navigate to:
- **Main App**: http://localhost:8000/static/index.html
- **About**: http://localhost:8000/static/about.html
- **Docs**: http://localhost:8000/static/docs.html
- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)

---

## API Documentation

### REST Endpoints

#### GET /api/status
Get system status and model information.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "num_checkpoints": 5,
  "num_examples": 20,
  "device": "cuda",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### GET /api/models/list
List available model checkpoints.

**Response:**
```json
{
  "checkpoints": [
    {
      "name": "step=10000",
      "step": 10000,
      "epoch": null,
      "file_size_mb": 250.5
    }
  ]
}
```

#### POST /api/models/load
Load a specific checkpoint.

**Request:**
```json
{
  "checkpoint_name": "step=10000"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Loaded checkpoint: step=10000",
  "model_info": {
    "loaded": true,
    "checkpoint": "step=10000",
    "total_parameters": 50000000,
    "parameters_millions": 50.0
  }
}
```

#### POST /api/generate
Start a generation job.

**Request:**
```json
{
  "mode": "unconditional",
  "generator_type": "mar",
  "scan_order": "row_major",
  "length": 256,
  "temperature": 1.0,
  "cfg": 1.0,
  "create_gif": true,
  "show_progress": true,
  "show_grid": false
}
```

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "pending",
  "message": "Generation job started"
}
```

#### GET /api/generate/{job_id}
Check generation job status.

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "progress": 1.0,
  "midi_url": "/outputs/uuid-here/output.mid",
  "image_url": "/outputs/uuid-here/output.png",
  "gif_url": "/outputs/uuid-here/output.gif"
}
```

### WebSocket Endpoint

#### WS /ws/generate
Real-time generation with progress updates.

**Client sends:**
```json
{
  "mode": "unconditional",
  "length": 256,
  "temperature": 1.0
}
```

**Server streams:**
```json
{
  "status": "running",
  "progress": 0.5,
  "message": "Generating..."
}
```

**Final message:**
```json
{
  "status": "completed",
  "progress": 1.0,
  "midi_url": "/outputs/uuid/output.mid",
  "image_url": "/outputs/uuid/output.png",
  "gif_url": "/outputs/uuid/output.gif"
}
```

---

## Frontend Features

### 1. Model Configuration Panel
- Select generator type (MAR/AR)
- Choose scan order for AR (row-major/column-major)
- Load different checkpoints
- View model information

### 2. Generation Mode Selection
- **Unconditional**: Generate from scratch
- **Conditional**: Continue from a given example
- **Inpainting**: Fill in masked regions

### 3. Interactive Piano Roll Canvas
- Zoom in/out
- Pan with mouse drag
- Real-time rendering
- Colormap visualization

### 4. Advanced Parameters
- Temperature control
- CFG scale adjustment
- Visualization options (GIF, progress, grid)

### 5. Results Display
- Generated piano roll image
- Download MIDI, image, GIF
- Play MIDI (future feature)

### 6. Fractal Loading Animation
- Mandelbrot-like animation during model loading
- Visual feedback for long operations

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f fractalmidi

# Stop the services
docker-compose down
```

### Configuration

Edit `docker-compose.yml` to customize:
- Port mappings
- Volume mounts
- GPU allocation
- Environment variables

### With Nginx Reverse Proxy

The `docker-compose.yml` includes an optional nginx service for:
- Static file serving
- Load balancing
- SSL termination (configure separately)

---

## TouchDesigner Integration

See `web/TOUCHDESIGNER_INTEGRATION.md` for detailed guide.

**Quick Start:**

1. Use WebSocket to connect to `ws://localhost:8000/ws/generate`
2. Send generation requests as JSON
3. Receive real-time updates
4. Load generated MIDI/images for visualization

**Example TouchDesigner Script:**
```python
import json

# Send generation request
request = {
    "mode": "unconditional",
    "length": 256,
    "temperature": 1.0
}
op('webclient1').sendText(json.dumps(request))

# Handle response
def onReceiveText(dat, text):
    data = json.loads(text)
    if data.get('status') == 'completed':
        # Load generated image
        op('moviefilein1').par.file = data['image_url']
```

---

## Development

### Project Structure

- **Backend**: FastAPI application with async support
- **Frontend**: Vanilla JavaScript (no framework dependencies)
- **Communication**: REST API + WebSocket
- **Styling**: Custom CSS with modern design

### Adding New Features

1. **Backend**: Add endpoints in `app.py`, implement logic in separate modules
2. **Frontend**: Add UI in `index.html`, implement logic in `main.js`
3. **Styling**: Update `style.css`

### Code Style

- Python: Follow PEP 8
- JavaScript: Use ES6+ features
- CSS: Use CSS variables for theming

---

## Troubleshooting

### Model Loading Issues

**Problem**: "No checkpoints available"
**Solution**: Ensure `.ckpt` files are in `outputs/checkpoints/`

**Problem**: "Failed to load checkpoint"
**Solution**: Check checkpoint compatibility and CUDA availability

### Generation Errors

**Problem**: "No model loaded"
**Solution**: Load a checkpoint first via the UI or API

**Problem**: "Generation timeout"
**Solution**: Increase timeout in `config.py` or reduce generation length

### WebSocket Issues

**Problem**: WebSocket connection fails
**Solution**: Check firewall settings, ensure server is running

**Problem**: Progress updates not received
**Solution**: Check browser console for errors, verify WebSocket URL

### Performance Issues

**Problem**: Slow generation
**Solution**: Use GPU, reduce length, lower num_iter_list

**Problem**: High memory usage
**Solution**: Reduce batch size, clear old generations

---

## Configuration

Edit `web/backend/config.py` to customize:

```python
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

@dataclass
class ModelConfig:
    checkpoint_dir: Path = Path("outputs/checkpoints")
    device: str = "cuda"  # or "cpu"
    max_batch_size: int = 4

@dataclass
class GenerationConfig:
    max_length: int = 512
    default_length: int = 256
    create_gif: bool = True
    gif_fps: int = 24
```

---

## Security Considerations

For production deployment:

1. **Enable HTTPS**: Configure SSL certificates in nginx
2. **Add Authentication**: Implement user authentication
3. **Rate Limiting**: Limit API requests per user
4. **Input Validation**: Validate all user inputs
5. **CORS**: Restrict allowed origins in production

---

## Performance Optimization

1. **Model Caching**: Keep model loaded in memory
2. **Result Caching**: Cache generated results
3. **Async Processing**: Use background tasks for generation
4. **CDN**: Serve static files via CDN
5. **Database**: Use database for job tracking (future)

---

## Future Enhancements

- [ ] User authentication and accounts
- [ ] Generation history and favorites
- [ ] MIDI playback in browser
- [ ] Real-time collaborative editing
- [ ] Mobile app (PWA)
- [ ] Advanced inpainting tools
- [ ] Style transfer
- [ ] Community gallery

---

## Support

For issues, questions, or contributions:
- Check the main FractalMIDI documentation
- Review API documentation at `/docs`
- Check logs in `logs/` directory

---

## License

See main FractalMIDI repository for license information.

---

## Acknowledgments

- FastAPI for the excellent web framework
- TouchDesigner for interactive art capabilities
- The FractalGen paper for the core architecture

---

**Enjoy creating music with FractalMIDI! 🎵✨**

