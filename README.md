# FractalMIDI 2.0: Tempo-Aware Hierarchical Music Generation

> *"FractalMIDI is not just a faster MIDI generator. It is a study on the self-similarity of musical time, exploring how AI can hallucinate structure from chaos in a way that mirrors human cognition."*

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FlashAttention](https://img.shields.io/badge/FlashAttention-2-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**FractalMIDI 2.0** is a state-of-the-art generative music system that treats music composition as a **hierarchical process**. Unlike traditional Autoregressive (AR) models (like Music Transformer) or flat Diffusion models that generate tokens sequentially, FractalMIDI first hallucinates a **Structure** (Tempo, Density) and then recursively "unfolds" this structure into **Content** (Notes, Velocities).

This "Fractal Unfolding" approach enables:
1.  **Massive Speedup**: $O(\log L)$ time complexity using parallel generation and FlashAttention-2.
2.  **Musical Coherence**: Strict adherence to global tempo and density curves via **Harmonic Compression** and **Flexible Bar Embeddings**.
3.  **Artistic Control**: Real-time interaction via WebUI and TouchDesigner (OSC).

---

## 🔥 Key Innovations (The "Why")

### 1. Structure vs. Content Paradigm
Most AI music models fail to maintain long-term structure because they treat a whole song as a flat sequence of notes.
*   **Level 0 (Macro)**: FractalMIDI generates a **Density Map** (how busy the music is) and a **Tempo Curve** (pacing) first.
*   **Level 1+ (Micro)**: It then generates the concrete **Notes** and **Velocities**, conditioned strictly on the Level 0 skeleton via **Harmonic Compression**.

### 2. Pitch-Aware Architecture & Flexible Bar Embedding
Standard CNNs treat the Pitch axis (Frequency) the same as the Time axis. This is physically incorrect.
*   **Innovation 1**: We use a specialized **2D Pitch-Sweep Convolution** (`kernel=[1, 12]`) that explicitly captures harmonic intervals (octaves, fifths).
*   **Innovation 2**: We introduce explicit **Flexible Bar Position Embeddings** ($E_{bar}$) so the model inherently "feels" the beat (0-max_bar_len position within a bar), eliminating rhythmic drift. This is now configurable to support various time signatures.

### 3. Configurable Deep Sampling
Instead of a single pass, FractalMIDI uses a **Refinement Loop**. After the initial generation, it randomly masks a configurable percentage (default 30%) of the tokens and re-predicts them, polishing the micro-details and ensuring smoother transitions.

### 4. FlashAttention-2 Integration
We replaced standard $O(L^2)$ Attention with **FlashAttention-2**, achieving linear memory scaling.
*   **Result**: Generate full 3-minute tracks (4096+ tokens) in **0.03 seconds** (vs. 5.5s for AR baselines).

---

## 🛠️ Installation

### Prerequisites
*   Linux / WSL2
*   NVIDIA GPU (CUDA 11.8+ recommended for FlashAttention)
*   Python 3.10+

### Setup Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/feiyueh/FractalMIDI.git
    cd FractalMIDI
    ```

2.  **Create Conda Environment**
    ```bash
    conda create -n frac python=3.10
    conda activate frac
    ```

3.  **Install Dependencies**
    It is critical to install PyTorch first, then FlashAttention.
    ```bash
    # Install PyTorch (adjust for your CUDA version)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install FlashAttention (requires nvcc)
    pip install flash-attn --no-build-isolation

    # Install other requirements
    pip install -r requirements.txt
    
    # Install web requirements
    pip install -r src/web/requirements_web.txt
    ```

---

## 🚆 Training

FractalMIDI employs a **Two-Stage Warmup** strategy to stabilize the hybrid classification/regression loss landscape.

### 1. Training Stages
*   **Phase 1 (0 - 5k steps)**: "Learn to Speak". The `structure_weight` is set to 0.0. The model focuses purely on predicting note placement (BCE Loss) and velocity (MSE Loss), establishing a basic musical vocabulary.
*   **Phase 2 (5k+ steps)**: "Learn to Flow". The `structure_weight` ramps up to 1.0. The model learns to co-generate valid tempo/density curves and adhere to them.

### 2. Key Hyperparameters
| Parameter | Value | Description |
|:---|:---|:---|
| `learning_rate` | 1e-4 | AdamW optimizer |
| `betas` | (0.9, 0.95) | Optimizer momentum |
| `batch_size` | 16 | Per GPU (adjust based on VRAM) |
| `weight_decay` | 0.01 | Regularization |
| `precision` | bf16 | Brain Float 16 mixed precision |
| `max_bar_len` | 16 | Standard 4/4 time signature (configurable) |

### 3. Run Training
To start training with the default configuration:

```bash
bash run_training.sh config/fractal_midi_v2.yaml
```

Or run directly with python:
```bash
python main.py --config config/fractal_midi_v2.yaml
    ```

---

## 🚀 Usage Guide

### 1. Web Interface (The Full Experience)
The WebUI offers the complete interactive suite: Fractal Visualizer, Inpainting Canvas, and Parameter Controls.

```bash
python src/web/backend/app.py
```
*   Open **http://localhost:8000** in your browser.
*   **Controls**: Adjust `Temperature` (Chaos) and `CFG` (Adherence).
*   **Inpainting**: Drag to select a region on the piano roll to regenerate.
*   **Visualizations**: See real-time "Condition Influence" to understand what the model is paying attention to.

### 2. Command Line Inference
For batch generation or headless servers.

**Unconditional Generation with Evaluation:**
```bash
python src/inference/inference.py \
    --generation_length 64 \
    --mode unconditional \
    --output_dir ./outputs \
    --evaluate
```

**Inpainting (Time Masking):**
Regenerate bars 16-32 of an existing MIDI file.
```bash
python src/inference/inference.py \
    --mode inpainting \
    --checkpoint path/to/model.ckpt \
    --input_midi input_track.mid \
    --mask_ranges 16-32
```

### 3. TouchDesigner Integration (OSC)
Turn FractalMIDI into a live performance instrument. The system listens for OSC messages on port `5005`.

**Setup:**
1.  Start the backend: `python src/web/backend/app.py`
2.  In TouchDesigner, create an **OSC Out** CHOP connected to `127.0.0.1:5005`.

**Supported Commands:**
| Address | Type | Description |
|:--------|:-----|:------------|
| `/fractal/param/temperature` | Float | Set sampling temperature (0.1 - 2.0) |
| `/fractal/param/cfg` | Float | Set classifier-free guidance scale |
| `/fractal/latent/level0` | List[Float] | Inject a custom latent vector to shape Level 0 structure |
| `/fractal/generate` | Bang | Trigger a new generation cycle |

---

## 📊 Benchmarks & Evaluation

We benchmarked FractalMIDI against a standard Autoregressive (AR) Transformer baseline on an NVIDIA RTX 4090.

| Sequence Length (Tokens) | FractalMIDI (s) | AR Baseline (s) | Speedup |
|:-------------------------|:----------------|:----------------|:--------|
| **256** (~10s)           | 0.034s          | 0.321s          | **9.4x**    |
| **1024** (~40s)          | 0.034s          | 1.388s          | **40.7x**   |
| **4096** (~3m)           | 0.035s          | 5.552s          | **160x**    |

*Note: FractalMIDI's inference time is effectively constant ($O(\log L)$) due to its parallel hierarchical nature, whereas AR models scale linearly ($O(L)$).*

### Objective Metrics
We now support automatic calculation of:
- **Pitch Entropy**: Measuring harmonic complexity.
- **Scale Consistency**: How well notes fit a key.
- **Groove Consistency**: Rhythmic stability over time.

---

## 📂 Project Structure

```
FractalMIDI/
├── ALGORITHM.md            # Technical whitepaper (READ THIS!)
├── main.py                 # Main entry point for training
├── src/                    # Source code
│   ├── dataset/            # Data pipeline (Density/Tempo extraction, Bar Pos)
│   ├── inference/          # CLI inference script & Metrics
│   ├── training/           # Training loop with Loss Warmup
│   ├── models/             # Model definitions
│   │   ├── temporal_fractal.py # Core TFN Architecture
│   │   ├── components.py       # Reusable blocks (AdaLN, Embeddings)
│   │   └── model_config.py     # Configuration dataclasses
│   └── web/
│       ├── backend/        # FastAPI & OSC Server
│       └── frontend/       # React UI (Source)
└── config/                 # Configuration files
```

---

## 🎓 Credits & Acknowledgements

**Author**: feiyueh

Special thanks to **"The Professor"** for the critical feedback regarding:
*   The misuse of 2D convolutions on 1D time series.
*   The necessity of separating Structure (Level 0) from Content (Level 1).
*   The importance of algorithmic efficiency ($O(\log L)$) for academic publication.
*   The need for flexible time signatures and rigorous ablation studies.

For a deep dive into the mathematics and architectural decisions, please refer to **[ALGORITHM.md](ALGORITHM.md)**.
