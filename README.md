# FractalMIDI

**FractalMIDI** is a hierarchical generative model for symbolic music (MIDI), inspired by [FractalGen](https://arxiv.org/abs/2401.06322). It treats music generation as a fractal process, starting from a coarse structural plan (Tempo/Density) and recursively refining it into detailed notes and velocities.

## Features

*   **Hierarchical Generation**: Generates music in 4 levels of resolution (Structure $\to$ Phrase $\to$ Beat $\to$ Content).
*   **Hybrid Architecture**: Combines **MAR** (Masked Auto-Regressive) for global structure planning with **AR** (Auto-Regressive) for detailed content generation.
*   **Pitch-wise LSTM Decoder**: Solves the "chord coherence" problem in AR models by generating notes sequentially within each time step.
*   **Structural Guidance (CFG)**: Uses Classifier-Free Guidance to enforce strict adherence to the generated structure without needing external labels.

## Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yaml
conda activate fractalmidi
```

### 2. Training
Train the model using the recommended hybrid configuration:
```bash
python main.py --config config/fractal_midi_v2.yaml
```

### 3. Inference
Generate music from a trained checkpoint:
```bash
python src/inference/inference.py --checkpoint outputs/pop909_hybrid/checkpoints/last.ckpt --num_samples 4
```

## Architecture Overview

| Level | Strategy | Description |
| :--- | :--- | :--- |
| **L0 (Structure)** | **MAR** | Generates the global Density and Tempo map (32 steps). |
| **L1 (Phrase)** | **MAR** | Refines structure to phrase level (64 steps). |
| **L2 (Beat)** | **AR** | Generates note content at beat level (128 steps). |
| **L3 (Content)** | **AR** | Final note generation at 1/16 note resolution (512 steps). |

See [ALGORITHM.md](ALGORITHM.md) for detailed technical explanation.

## Configuration

Modify `config/fractal_midi_v2.yaml` to experiment with different architectures:

*   **`seq_len_list`**: Define the sequence length for each level.
*   **`generator_type_list`**: Choose `['ar']` or `['mar']` for each level.
*   **`full_mask_prob`**: Probability of training generation from scratch (vs. inpainting).

## License
MIT
