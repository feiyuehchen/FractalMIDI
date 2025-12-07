from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict, Optional

@dataclass
class MetricsConfig:
    """Configuration for objective evaluation metrics."""
    pitch_entropy_bins: int = 128
    scale_consistency_threshold: float = 0.8
    groove_quantization: int = 16  # 16th notes

class MusicMetrics:
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()

    def calculate_pitch_entropy(self, piano_roll: np.ndarray) -> float:
        """
        Calculate Shannon entropy of pitch usage.
        piano_roll: (T, 128) or (2, T, 128)
        """
        if piano_roll.ndim == 3:
            # Use Note channel (index 0)
            activity = piano_roll[0] # (T, 128)
        else:
            activity = piano_roll

        # Sum over time -> histogram of pitch counts
        pitch_counts = np.sum(activity > 0.5, axis=0) # (128,)
        total_notes = np.sum(pitch_counts)
        
        if total_notes == 0:
            return 0.0
            
        probs = pitch_counts / total_notes
        # Remove zeros for log
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        # Normalize by max possible entropy (log2(128) = 7)? 
        # Or just return raw bits.
        return float(entropy)

    def calculate_scale_consistency(self, piano_roll: np.ndarray) -> float:
        """
        Estimate the fraction of notes that fit into the most likely scale.
        Simple heuristic: check major/minor scales for all 12 keys.
        """
        if piano_roll.ndim == 3:
            activity = piano_roll[0]
        else:
            activity = piano_roll
            
        # Collapse time
        chroma = np.sum(activity > 0.5, axis=0) # (128,)
        # Fold to 12 pitch classes
        chroma_12 = np.zeros(12)
        for i in range(128):
            chroma_12[i % 12] += chroma[i]
            
        total_notes = np.sum(chroma_12)
        if total_notes == 0:
            return 0.0
            
        # Define scale masks (Major: 2,2,1,2,2,2,1)
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10] # Natural minor
        
        max_in_scale = 0
        
        for root in range(12):
            # Check Major
            mask = np.zeros(12, dtype=bool)
            for interval in major_intervals:
                mask[(root + interval) % 12] = True
            in_scale_count = np.sum(chroma_12[mask])
            max_in_scale = max(max_in_scale, in_scale_count)
            
            # Check Minor
            mask = np.zeros(12, dtype=bool)
            for interval in minor_intervals:
                mask[(root + interval) % 12] = True
            in_scale_count = np.sum(chroma_12[mask])
            max_in_scale = max(max_in_scale, in_scale_count)
            
        return float(max_in_scale / total_notes)

    def calculate_groove_consistency(self, piano_roll: np.ndarray) -> float:
        """
        Measure how consistent the rhythm is. 
        Simple metric: Variance of onset positions modulo beat?
        Or: Ratio of notes on strong beats vs weak beats.
        Let's use onset density variance across bars.
        """
        if piano_roll.ndim == 3:
            activity = piano_roll[0]
        else:
            activity = piano_roll
            
        T = activity.shape[0]
        # Detect onsets: current > 0.5 and prev < 0.5
        # (Simplified)
        onsets = (activity > 0.5).astype(float)
        # Just count active notes per 16-step bar
        bar_len = 16
        num_bars = T // bar_len
        if num_bars < 2:
            return 1.0 # Too short
            
        onset_counts = []
        for i in range(num_bars):
            bar_slice = onsets[i*bar_len : (i+1)*bar_len, :]
            count = np.sum(bar_slice)
            onset_counts.append(count)
            
        # Consistency = 1 / (Coefficient of Variation of note counts + 1)
        # If notes per bar is constant, CV=0, Consistency=1.
        mean = np.mean(onset_counts)
        if mean == 0:
            return 0.0
        std = np.std(onset_counts)
        cov = std / mean
        return float(1.0 / (cov + 1.0))

    def evaluate_batch(self, piano_rolls: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a batch of piano rolls and return average metrics.
        """
        results = {
            'pitch_entropy': [],
            'scale_consistency': [],
            'groove_consistency': []
        }
        
        for pr in piano_rolls:
            results['pitch_entropy'].append(self.calculate_pitch_entropy(pr))
            results['scale_consistency'].append(self.calculate_scale_consistency(pr))
            results['groove_consistency'].append(self.calculate_groove_consistency(pr))
            
        return {k: float(np.mean(v)) for k, v in results.items()}

