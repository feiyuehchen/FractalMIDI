from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict, Optional
import math

@dataclass
class MetricsConfig:
    """Configuration for objective evaluation metrics."""
    pitch_entropy_bins: int = 12
    scale_consistency_threshold: float = 0.8
    groove_quantization: int = 16  # 16th notes per bar

class MusicMetrics:
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()

    def calculate_pitch_class_entropy(self, piano_roll: np.ndarray) -> float:
        """
        Calculate Shannon entropy of pitch class usage (0-11).
        MusDr Metric: H
        """
        if piano_roll.ndim == 3:
            activity = piano_roll[0] # (T, 128)
        else:
            activity = piano_roll

        # Sum over time -> histogram of pitch counts
        pitch_counts = np.sum(activity > 0.5, axis=0) # (128,)
        
        # Fold to 12 pitch classes
        chroma_counts = np.zeros(12)
        for i in range(128):
            chroma_counts[i % 12] += pitch_counts[i]
            
        total_notes = np.sum(chroma_counts)
        if total_notes == 0:
            return 0.0
            
        probs = chroma_counts / total_notes
        # Remove zeros
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def calculate_groove_similarity(self, piano_roll: np.ndarray) -> float:
        """
        Grooving Pattern Similarity (GS).
        Pairwise similarity of rhythm vectors between bars.
        """
        if piano_roll.ndim == 3:
            activity = piano_roll[0]
        else:
            activity = piano_roll
            
        T = activity.shape[0]
        bar_len = 16 # Assuming 16th note resolution
        num_bars = T // bar_len
        
        if num_bars < 2:
            return 0.0
            
        # Extract onset vectors per bar
        # Collapse pitch: just check if ANY note is played at time t
        # (T,)
        onsets = np.max(activity > 0.5, axis=1).astype(float)
        
        bar_vectors = []
        for i in range(num_bars):
            vec = onsets[i*bar_len : (i+1)*bar_len]
            if np.sum(vec) > 0:
                bar_vectors.append(vec)
        
        if len(bar_vectors) < 2:
            return 0.0
            
        bar_vectors = np.stack(bar_vectors) # (N, 16)
        
        # Compute pairwise cosine similarity
        # Normalize vectors
        norms = np.linalg.norm(bar_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized = bar_vectors / norms
        
        sim_matrix = np.matmul(normalized, normalized.T) # (N, N)
        
        # Average of upper triangle (excluding diagonal)
        tri_idx = np.triu_indices(len(bar_vectors), k=1)
        if len(tri_idx[0]) == 0:
            return 0.0
            
        gs = np.mean(sim_matrix[tri_idx])
        return float(gs)

    def calculate_structureness(self, piano_roll: np.ndarray) -> float:
        """
        Structureness Indicator (SI).
        Uses Self-Similarity Matrix (SSM) of chroma vectors.
        Simplified version: Average correlation of distant parts?
        Or fitness scape plot? 
        Let's implement a simple lag-based similarity (autocorrelation) to detect repetition.
        """
        if piano_roll.ndim == 3:
            activity = piano_roll[0]
        else:
            activity = piano_roll
            
        # Convert to Chroma Sequence
        T = activity.shape[0]
        chroma_seq = np.zeros((T, 12))
        for t in range(T):
            for p in range(128):
                if activity[t, p] > 0.5:
                    chroma_seq[t, p % 12] += 1
        
        # Normalize chroma vectors
        norms = np.linalg.norm(chroma_seq, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chroma_seq = chroma_seq / norms
        
        # Compute SSM (T, T) - downsampled to save time?
        # Let's downsample by 4 (beat level)
        chroma_down = chroma_seq[::4]
        if len(chroma_down) < 10:
            return 0.0
            
        ssm = np.matmul(chroma_down, chroma_down.T)
        
        # Check for diagonals (repetition)
        # We look at diagonals with offset > 0
        # High SI means high values in diagonals.
        # Simple metric: mean of top 10% values in upper triangle?
        
        # Better: Look for peaks in autocorrelation of the chroma sequence
        # (This is equivalent to summing diagonals of SSM)
        n = len(chroma_down)
        autocorr = np.zeros(n//2)
        for lag in range(1, n//2):
             # Correlation at lag
             # vec[0:n-lag] vs vec[lag:n]
             v1 = chroma_down[:n-lag]
             v2 = chroma_down[lag:]
             # Row-wise dot product then mean
             dot = np.sum(v1 * v2, axis=1)
             autocorr[lag] = np.mean(dot)
             
        # Peak of autocorrelation (ignoring lag 0 which is 1.0)
        # We ignore very small lags (local continuity)
        if len(autocorr) > 4:
            si = np.max(autocorr[4:])
        else:
            si = 0.0
            
        return float(si)

    def evaluate_batch(self, piano_rolls: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a batch of piano rolls and return average metrics.
        """
        results = {
            'pitch_class_entropy': [],
            'groove_similarity': [],
            'structureness': []
        }
        
        for pr in piano_rolls:
            results['pitch_class_entropy'].append(self.calculate_pitch_class_entropy(pr))
            results['groove_similarity'].append(self.calculate_groove_similarity(pr))
            results['structureness'].append(self.calculate_structureness(pr))
            
        return {k: float(np.mean(v)) for k, v in results.items()}
