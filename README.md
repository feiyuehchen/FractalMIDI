# FractalMIDI 2.0

FractalMIDI is a hierarchical generative music system powered by the **Tempo-Aware Temporal Fractal Network (TFN)**. It generates high-quality multi-track MIDI data by first composing the musical structure (Tempo & Density) and then fleshing out the details (Notes & Velocity).

## Key Features

*   **Hierarchical Generation**: Generates structure (Level 0) before content (Level 1+), ensuring long-term coherence.
*   **Tempo-Aware**: Explicitly models and generates tempo curves, allowing for expressive timing variations.
*   **Inpainting**: Can fill in missing gaps in a MIDI file or extend existing motifs.
*   **Web Interface**: Full-featured WebUI for interactive generation, visualization, and integration with TouchDesigner.

## Algorithm
FractalMIDI 2.0 moves away from image-based Conv2d approaches to a physically-motivated 1D Time-Convolution architecture.
For a deep dive into the architecture, see [ALGORITHM.md](ALGORITHM.md).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training
Train the TFN model on your dataset (configured in `config/fractal_midi_v2.yaml`).

```bash
sh run_training.sh
```

### 2. Inference

**Unconditional Generation** (Generate from scratch):
```bash
python inference.py --mode unconditional --num_samples 4 --save_images
```

**Inpainting** (Fill in gaps or modify specific regions):
```bash
# Mask and regenerate measures 32-64 (in 16th notes)
python inference.py --mode inpainting \
    --input_midi input_song.mid \
    --mask_ranges 32-64 \
    --save_images
```

### 3. Web UI
Launch the interactive web interface:

```bash
python main.py
```
Access at `http://localhost:8000`.

## Project Structure

*   `models/`: TFN model definitions (`temporal_fractal.py`).
*   `dataset.py`: MIDI data pipeline handling Note, Velocity, Tempo, and Density.
*   `inference.py`: CLI for generation and inpainting.
*   `web/`: FastAPI backend and HTML/JS frontend.
*   `config/`: YAML configuration files.
