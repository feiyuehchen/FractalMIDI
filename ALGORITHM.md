# FractalMIDI Algorithm & Architecture

This document details the architecture, algorithms, and implementation details of FractalMIDI 2.0.

## 1. Core Architecture: Tempo-Aware Temporal Fractal Network (TFN)

FractalMIDI generates music through a **Hierarchical Fractal Structure**, similar to how fractal images are generated. The generation process moves from coarse structures (Tempo/Density) to fine details (Notes/Velocity).

### 1.1 Hierarchical Levels

The model processes music at multiple temporal resolutions. For a sequence length of 512 (32 bars), the recommended hierarchy is:

| Level | Resolution | Sequence Length | Meaning | Content Type | Generator |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L0** | 16 steps (~1 bar) | 32 | Global Structure | Density & Tempo | **MAR** (Recommended) |
| **L1** | 4 steps (1 beat) | 64 | Phrase Structure | Density & Tempo / Rough Notes | **MAR** |
| **L2** | 2 steps (1/8 note) | 128 | Rhythm & Harmony | Notes & Velocity | **AR** |
| **L3** | 1 step (1/16 note) | 512 | Fine Details | Notes & Velocity | **AR** |

### 1.2 Hybrid Generation Strategy (MAR + AR)

FractalMIDI supports a hybrid generation strategy, combining the strengths of two paradigms:

1.  **MAR (Masked Auto-Regressive) for Structure (Levels 0-1)**:
    *   **Role**: Acts as the "Architect".
    *   **Mechanism**: Uses bidirectional attention to see the entire song duration at once. It iteratively refines a blurry "plan" (density/tempo map) into a clear structure.
    *   **Advantage**: Ensures global coherence (e.g., consistent song structure, proper ending) and allows for flexible inpainting.

2.  **AR (Auto-Regressive) for Content (Levels 2-3)**:
    *   **Role**: Acts as the "Performer".
    *   **Mechanism**: Generates music strictly from left to right (time $t \to t+1$), conditioned on the structure plan provided by MAR.
    *   **Advantage**: Produces coherent, flowing melodies and harmonically correct chords.

## 2. Key Components

### 2.1 Pitch-wise LSTM Decoder (for AR Layers)

Standard Transformer AR models often struggle with "intra-chord coherence" (generating C-E-G together correctly) because they predict all pitches at time $t$ simultaneously.

FractalMIDI introduces a specialized **LSTM Output Projector** for AR layers:
*   **Time-wise AR**: The Transformer predicts a latent embedding $h_t$ for time step $t$.
*   **Pitch-wise AR**: A 2-layer LSTM takes $h_t$ as a condition and generates the 128 pitches (MIDI 0-127) **sequentially**.
    *   $P(pitch_i | h_t, pitch_{<i})$
*   **Result**: Structurally solid chords and polyphony.

### 2.2 Classifier-Free Guidance (CFG) via Structure

We implement CFG without external class labels by using the **Previous Level's Structure** as the condition.

*   **Conditional Pass**: Generate layer $L$ using structure from $L-1$ as guidance.
*   **Unconditional Pass**: Generate layer $L$ with no guidance (cond = None).
*   **Guidance**: $Logits = Logits_{uncond} + w \cdot (Logits_{cond} - Logits_{uncond})$
*   **Effect**: Increasing $w$ (>1.0) forces the detailed notes to strictly adhere to the density/tempo plan, reducing "noodling" or structural drift.

## 3. Algorithms

### 3.1 Training

*   **MAR Layers**:
    *   Input: Randomly masked sequence.
    *   Loss: Reconstruction error of masked tokens.
*   **AR Layers**:
    *   Input: Full sequence shifted by 1 (Teacher Forcing).
    *   Loss: Prediction error of next time step.
    *   **LSTM Training**: Parallelized via Teacher Forcing (all pitches known during training).

### 3.2 Sampling (Inference)

1.  **L0 Generation (MAR)**: Start with noise, iteratively refine density/tempo map.
2.  **L1 Generation (MAR)**: Upsample L0 output, use as condition to generate L1 structure.
3.  **L2 Generation (AR)**:
    *   Upsample L1 output as condition.
    *   Loop $t = 0 \to T$:
        *   Transformer predicts $h_t$.
        *   LSTM generates 128 pitches for $h_t$.
4.  **L3 Generation (AR)**:
    *   Repeat AR process at finest resolution.

## 4. Configuration Guide

To configure the recommended hybrid model (4 layers, MAR->AR):

```yaml
model:
  architecture:
    seq_len_list: [32, 64, 128, 512]  # Resolution hierarchy
    embed_dim_list: [512, 512, 256, 128]
  generator:
    generator_type_list: [mar, mar, ar, ar] # Hybrid strategy
```
