"""
Example Manager for managing validation set MIDI examples.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import midi_to_piano_roll
from visualizer import piano_roll_to_image
import symusic

logger = logging.getLogger(__name__)


@dataclass
class ExampleInfo:
    """Information about a MIDI example."""
    id: str
    name: str
    file_path: str
    duration_seconds: float
    num_notes: int
    time_steps: int
    pitch_range: tuple  # (min_pitch, max_pitch)
    thumbnail_path: Optional[str] = None
    tags: List[str] = None


class ExampleManager:
    """
    Manages validation set MIDI examples for conditional and inpainting generation.
    """
    
    def __init__(self, examples_dir: Path, max_examples: int = 100):
        """
        Initialize example manager.
        
        Args:
            examples_dir: Directory containing MIDI examples
            max_examples: Maximum number of examples to load
        """
        self.examples_dir = Path(examples_dir)
        self.max_examples = max_examples
        self.examples: Dict[str, ExampleInfo] = {}
        self.metadata_file = self.examples_dir / "examples_metadata.json"
        
        # Create directory if it doesn't exist
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self._load_or_create_metadata()
    
    def _load_or_create_metadata(self):
        """Load metadata from file or create it by scanning MIDI files."""
        if self.metadata_file.exists():
            logger.info("Loading examples metadata from file")
            self._load_metadata()
        else:
            logger.info("Creating examples metadata by scanning MIDI files")
            self._create_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                example = ExampleInfo(**item)
                self.examples[example.id] = example
            
            logger.info(f"Loaded {len(self.examples)} examples from metadata")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self._create_metadata()
    
    def _create_metadata(self):
        """Create metadata by scanning MIDI files in examples directory."""
        logger.info(f"Scanning for MIDI files in {self.examples_dir}")
        
        midi_files = list(self.examples_dir.glob("*.mid")) + list(self.examples_dir.glob("*.midi"))
        midi_files = midi_files[:self.max_examples]
        
        logger.info(f"Found {len(midi_files)} MIDI files")
        
        for i, midi_path in enumerate(midi_files):
            try:
                # Load MIDI using symusic directly to avoid method signature issues
                # score = symusic.Score(str(midi_path))
                
                # Convert to piano roll using our standalone helper or re-implement logic here
                # to be safe, we re-implement the core logic to avoid any library version mismatches
                
                # 1. Load Score
                try:
                    score = symusic.Score(str(midi_path))
                except TypeError:
                    # Fallback for different symusic versions if needed, but usually path string works
                    score = symusic.Score(str(midi_path))
                
                # 2. Basic stats
                ticks_per_quarter = score.ticks_per_quarter
                ticks_per_16th = ticks_per_quarter // 4
                
                # 3. Convert to piano roll manually to ensure control
                total_ticks = 0
                for track in score.tracks:
                    if track.is_drum: continue
                    for note in track.notes:
                        total_ticks = max(total_ticks, note.end)
                
                if total_ticks == 0: total_ticks = ticks_per_16th * 16
                
                length = (total_ticks + ticks_per_16th - 1) // ticks_per_16th
                
                # Initialize piano roll
                piano_roll = np.zeros((128, length), dtype=np.float32)
                
                # Fill
                for track in score.tracks:
                    if track.is_drum: continue
                    for note in track.notes:
                        start = note.time // ticks_per_16th
                        end = note.end // ticks_per_16th
                        if start >= length: continue
                        end = min(end, length)
                        val = note.velocity / 127.0
                        piano_roll[note.pitch, start:end] = np.maximum(piano_roll[note.pitch, start:end], val)

                # Extract info
                time_steps = piano_roll.shape[1]
                duration_seconds = time_steps / 16.0 * (60 / 120) # Approx assuming 120bpm if unknown, or use real time
                # Better: symusic gives real time usually? 
                # Let's stick to steps for now as that's what matters for model
                
                num_notes = int((piano_roll > 0).sum())
                
                # Find pitch range
                pitch_indices = np.where(piano_roll.sum(axis=1) > 0)[0]
                if len(pitch_indices) > 0:
                    min_pitch = int(pitch_indices.min())
                    max_pitch = int(pitch_indices.max())
                else:
                    min_pitch = 0
                    max_pitch = 127
                
                # Create example ID
                example_id = f"ex_{i:04d}"
                
                # Create thumbnail
                thumbnail_path = self.examples_dir / f"{example_id}_thumb.png"
                self._create_thumbnail(piano_roll, thumbnail_path)
                
                # Create example info
                example = ExampleInfo(
                    id=example_id,
                    name=midi_path.stem,
                    file_path=str(midi_path.relative_to(self.examples_dir)),
                    duration_seconds=duration_seconds,
                    num_notes=num_notes,
                    time_steps=time_steps,
                    pitch_range=(min_pitch, max_pitch),
                    thumbnail_path=str(thumbnail_path.relative_to(self.examples_dir)),
                    tags=[]
                )
                
                self.examples[example_id] = example
                logger.info(f"Processed {i+1}/{len(midi_files)}: {midi_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {midi_path}: {e}")
        
        # Save metadata
        self._save_metadata()
        logger.info(f"Created metadata for {len(self.examples)} examples")
    
    def _create_thumbnail(self, piano_roll: np.ndarray, output_path: Path):
        """Create thumbnail image for a piano roll."""
        try:
            import torch
            
            # Crop to reasonable length for thumbnail
            max_length = 256
            if piano_roll.shape[1] > max_length:
                piano_roll = piano_roll[:, :max_length]
            
            # Convert to image
            img = piano_roll_to_image(
                torch.from_numpy(piano_roll),
                apply_colormap=True,
                return_pil=True,
                min_height=128
            )
            
            # Resize to thumbnail size
            img.thumbnail((256, 128), Image.LANCZOS)
            
            # Save
            img.save(output_path)
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            data = [asdict(example) for example in self.examples.values()]
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def list_examples(self, limit: Optional[int] = None) -> List[ExampleInfo]:
        """
        List all examples.
        
        Args:
            limit: Maximum number of examples to return
            
        Returns:
            List of ExampleInfo objects
        """
        examples = list(self.examples.values())
        if limit is not None:
            examples = examples[:limit]
        return examples
    
    def get_example(self, example_id: str) -> Optional[ExampleInfo]:
        """
        Get example by ID.
        
        Args:
            example_id: Example ID
            
        Returns:
            ExampleInfo or None if not found
        """
        return self.examples.get(example_id)
    
    def load_example_piano_roll(self, example_id: str, target_length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Load piano roll for an example.
        
        Args:
            example_id: Example ID
            target_length: Optional target length (will crop or pad)
            
        Returns:
            Piano roll numpy array (128, T) or None if not found
        """
        example = self.get_example(example_id)
        if example is None:
            return None
        
        try:
            midi_path = self.examples_dir / example.file_path
            # Directly use midi_to_piano_roll with string path to avoid symusic type errors
            piano_roll = midi_to_piano_roll(str(midi_path), target_length=target_length)
            return piano_roll
        except Exception as e:
            logger.error(f"Error loading piano roll for {example_id}: {e}")
            return None
    
    def search_examples(self, 
                       min_duration: Optional[float] = None,
                       max_duration: Optional[float] = None,
                       min_notes: Optional[int] = None,
                       max_notes: Optional[int] = None,
                       tags: Optional[List[str]] = None) -> List[ExampleInfo]:
        """
        Search examples by criteria.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_notes: Minimum number of notes
            max_notes: Maximum number of notes
            tags: List of tags to filter by
            
        Returns:
            List of matching ExampleInfo objects
        """
        results = []
        
        for example in self.examples.values():
            # Check duration
            if min_duration is not None and example.duration_seconds < min_duration:
                continue
            if max_duration is not None and example.duration_seconds > max_duration:
                continue
            
            # Check notes
            if min_notes is not None and example.num_notes < min_notes:
                continue
            if max_notes is not None and example.num_notes > max_notes:
                continue
            
            # Check tags
            if tags is not None and example.tags is not None:
                if not any(tag in example.tags for tag in tags):
                    continue
            
            results.append(example)
        
        return results

