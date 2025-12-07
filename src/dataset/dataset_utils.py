from dataclasses import dataclass
import numpy as np
import symusic

@dataclass
class ProcessingConfig:
    """Configuration for MIDI processing."""
    max_bar_len: int = 64  # Maximum length of a bar in 16th notes (supports up to 16/4 or similar)
    default_time_signature: tuple = (4, 4)

def compute_bar_pos(score: symusic.Score, ticks_per_16th: int, config: ProcessingConfig = None) -> np.ndarray:
    """
    Compute bar positions for each time step in the score.
    Handles time signature changes dynamically.
    
    Args:
        score: symusic.Score object
        ticks_per_16th: Number of ticks per 16th note
        config: ProcessingConfig object (optional)
        
    Returns:
        bar_pos: (T,) numpy array of integers, where T is total duration in 16th notes.
                 Values are 0-indexed positions within the current bar.
    """
    if config is None:
        config = ProcessingConfig()
        
    # Calculate total duration in steps
    max_end_time = 0
    for track in score.tracks:
        if len(track.notes) > 0:
            track_end = max(note.time + note.duration for note in track.notes)
            max_end_time = max(max_end_time, track_end)
            
    duration_steps = (max_end_time + ticks_per_16th - 1) // ticks_per_16th
    duration_steps = max(int(duration_steps), 1)
    
    bar_pos = np.zeros(duration_steps, dtype=np.int64)
    
    # Get time signatures and sort
    time_sigs = score.time_signatures
    if not time_sigs:
        # Default 4/4
        # 4/4 = 4 beats per bar, quarter note gets beat. 
        # 16th notes per bar = 4 * 4 = 16
        steps_per_bar = 16
        bar_pos[:] = np.arange(duration_steps) % steps_per_bar
        return bar_pos
        
    time_sigs.sort(key=lambda x: x.time)
    
    # Fill bar positions
    current_ts_idx = 0
    current_bar_start_step = 0
    current_steps_per_bar = 16 # Default
    
    # Initialize with first time signature if it starts at 0
    if time_sigs[0].time == 0:
        ts = time_sigs[0]
        # beats per bar * (16 / denominator) -> steps per bar
        # e.g. 3/4 -> 3 * (16/4) = 12 steps
        # e.g. 6/8 -> 6 * (16/8) = 12 steps
        current_steps_per_bar = int(ts.numerator * (16 / ts.denominator))
        current_ts_idx = 1
    
    for t in range(duration_steps):
        tick_time = t * ticks_per_16th
        
        # Check for new time signature
        if current_ts_idx < len(time_sigs) and time_sigs[current_ts_idx].time <= tick_time:
            ts = time_sigs[current_ts_idx]
            new_steps_per_bar = int(ts.numerator * (16 / ts.denominator))
            
            # If time signature changed, we might reset the bar phase?
            # Usually time signature changes happen at bar boundaries.
            # We assume the previous bar ended exactly at this new time signature.
            current_bar_start_step = t
            current_steps_per_bar = new_steps_per_bar
            current_ts_idx += 1
            
        # Calculate position in current bar
        pos_in_bar = (t - current_bar_start_step) % current_steps_per_bar
        
        # Cap at max_bar_len to prevent index errors in embedding
        bar_pos[t] = min(pos_in_bar, config.max_bar_len - 1)
        
    return bar_pos

