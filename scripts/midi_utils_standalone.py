import symusic
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use the standalone midi_to_piano_roll from inference.py
# This avoids circular imports and dependency on ExampleManager for basic logic
from inference import midi_to_piano_roll

def standalone_midi_to_piano_roll(midi_path, target_length=None):
    """Standalone wrapper to load MIDI and convert to piano roll."""
    try:
        return midi_to_piano_roll(midi_path, target_length=target_length)
    except Exception as e:
        print(f"Error in standalone_midi_to_piano_roll for {midi_path}: {e}")
        return None

