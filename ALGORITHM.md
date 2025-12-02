# Tempo-Aware Temporal Fractal Network (TFN)

FractalMIDI 2.0 introduces the **Tempo-Aware Temporal Fractal Network (TFN)**, a hierarchical generative model designed specifically for symbolic music (MIDI). This architecture addresses the limitations of treating music as 2D images (like ImageNet models) by acknowledging the physical distinctions between **Time** and **Pitch**.

## Core Philosophy: Structure vs. Content

Unlike standard "flat" generative models, TFN decomposes music generation into two distinct semantic levels:

1.  **Level 0: Structure (Macro-Dynamics)**
    *   **Resolution**: Coarse (e.g., 1/16 of full length).
    *   **Channels**:
        *   **Density**: The density of notes per time step (how "busy" the music is).
        *   **Tempo**: The speed of the music (BPM normalized).
    *   **Goal**: Establish the "skeleton" of the piece—its pacing and intensity—without worrying about specific notes.

2.  **Level 1+: Content (Micro-Details)**
    *   **Resolution**: Fine (Full resolution).
    *   **Channels**:
        *   **Note**: Binary On/Off status for each pitch (0-127).
        *   **Velocity**: Dynamics/Loudness of each note (0-1).
    *   **Goal**: "Flesh out" the structure with concrete musical notes, harmonies, and melodies, conditioned on the Level 0 skeleton.

## Architecture Details

### 1. 1D Temporal Convolutions (Not 2D)
Previous approaches often used 2D Convolutions (Time x Pitch), which incorrectly assume Pitch invariance is the same as Time invariance. TFN uses **1D convolutions over Time**, treating Pitch as a feature dimension.
*   **FractalInputProj**: Projects the (2, T, 128) Piano Roll into a (B, T, E) embedding space using a Linear projection over the flattened pitch dimension. This captures global pitch relationships (chords, intervals) at each time step.

### 2. Hierarchical Conditioning via AdaLN
Information flows from Level 0 to Level 1+ via **Adaptive Layer Normalization (AdaLN)**.
*   The Level 0 output (Structure) is upsampled and projected to modulate the features of the Level 1 generator.
*   This ensures that the generated notes strictly follow the established density and tempo curves.

### 3. Fractal Loss Function
The model is trained with a multi-objective loss:
*   **Structure Loss**: MSE on Density and Tempo curves.
*   **Content Loss**:
    *   **Note Loss**: Weighted Binary Cross Entropy (BCE) to handle class imbalance (sparse notes).
    *   **Velocity Loss**: MSE applied *only* where notes exist (masked regression).

## Inpainting & Iterative Refinement (MAR)

TFN uses **Masked Auto-Regressive (MAR)** generation logic, allowing for flexible inpainting:

*   **Iterative Sampling**: The model refines the output over multiple steps (e.g., 8 -> 4 -> 2 steps per level).
*   **Inpainting Constraint**:
    *   User provides `initial_content` (a partial MIDI file) and a `mask` (regions to regenerate).
    *   At every step of the iterative process, the "known" regions from the `initial_content` are forcibly injected back into the canvas.
    *   The model effectively "hallucinates" the missing parts to be musically consistent with the known parts and the hierarchical structure.

## Data Representation
*   **Input**: `(2, T, 128)` Tensor representing [Note On, Velocity].
*   **Auxiliary**: `(T,)` Tempo curve and `(T,)` Density curve derived during data loading.

