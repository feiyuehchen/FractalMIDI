import torch
from torch.utils.data import Dataset, Sampler
import symusic
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import random
import hashlib
import functools
from .dataset_utils import compute_bar_pos, ProcessingConfig

# ============================================================================
# Dataset Configuration (dataclasses)
# ============================================================================

@dataclass
class MIDIDatasetConfig:
    """Configuration for MIDI dataset."""
    file_list_path: str  # Path to txt file with MIDI file paths
    ticks_per_16th: int = 120  # Number of ticks per 1/16 note after quantization
    # Data augmentation: random crop settings
    random_crop: bool = True  # Enable random cropping for data augmentation
    crop_length: int = 256  # Target length in 16th notes (if random_crop=True)
    min_length: int = 64  # Minimum length for a valid crop
    augment_factor: int = 1  # How many random crops per MIDI file (effective dataset size multiplier)
    pad_to_multiple: int = 4  # Ensure output length is multiple of this value
    cache_in_memory: bool = True  # Cache piano rolls in RAM to avoid repeated decoding
    cache_dir: Optional[str] = None  # Optional directory for on-disk cache (npz)
    pitch_shift_range: Tuple[int, int] = (-3, 3)  # Inclusive semitone shift range for augmentation
    max_bar_len: int = 64 # Max length of bar for embedding
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.ticks_per_16th > 0, "ticks_per_16th must be positive"
        assert self.augment_factor >= 1, "augment_factor must be >= 1"
        assert self.pad_to_multiple > 0, "pad_to_multiple must be positive"
        min_shift, max_shift = self.pitch_shift_range
        assert min_shift <= max_shift, "pitch_shift_range must be (min_shift, max_shift) with min <= max"
        assert -128 <= min_shift <= 128 and -128 <= max_shift <= 128, "pitch_shift_range must stay within [-128, 128]"


@dataclass
class BucketSamplerConfig:
    """Configuration for bucket batch sampler."""
    batch_size: int = 4  # Batch size
    bucket_boundaries: Optional[List[int]] = None  # Duration boundaries for bucketing
    drop_last: bool = False  # Whether to drop the last incomplete batch
    shuffle: bool = True  # Whether to shuffle samples within each bucket
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        if self.bucket_boundaries is not None:
            assert all(b > 0 for b in self.bucket_boundaries), \
                "All bucket boundaries must be positive"
            assert self.bucket_boundaries == sorted(self.bucket_boundaries), \
                "Bucket boundaries must be sorted"


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    # Dataset config
    dataset: MIDIDatasetConfig = field(default_factory=lambda: MIDIDatasetConfig(
        file_list_path="data/lists/train.txt"
    ))
    
    # Sampler config (optional, only used if not using random_crop)
    sampler: BucketSamplerConfig = field(default_factory=BucketSamplerConfig)
    
    # DataLoader specific
    num_workers: int = 4  # Number of worker processes
    pin_memory: bool = True  # Pin memory for faster GPU transfers
    patch_size: int = 4  # Patch size for padding (must match model's patch_size)
    use_bucket_sampler: bool = False  # Use bucket sampler (only if not using random_crop)
    persistent_workers: bool = True  # Keep workers alive between epochs
    prefetch_factor: int = 2  # Number of batches prefetched per worker
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.patch_size > 0, "patch_size must be positive"
        if self.num_workers == 0:
            self.prefetch_factor = 0
        else:
            assert self.prefetch_factor >= 1, "prefetch_factor must be >= 1 when num_workers > 0"
    
    @classmethod
    def training_default(cls, file_list_path: str = "data/lists/train.txt", **kwargs):
        """Default configuration for training."""
        return cls(
            dataset=MIDIDatasetConfig(file_list_path=file_list_path),
            sampler=BucketSamplerConfig(
                batch_size=8,
                shuffle=True,
                drop_last=True
            ),
            num_workers=4,
            **kwargs
        )
    
    @classmethod
    def validation_default(cls, file_list_path: str = "data/lists/valid.txt", **kwargs):
        """Default configuration for validation."""
        return cls(
            dataset=MIDIDatasetConfig(file_list_path=file_list_path),
            sampler=BucketSamplerConfig(
                batch_size=8,
                shuffle=False,
                drop_last=False
            ),
            num_workers=4,
            **kwargs
        )
    
    @classmethod
    def test_default(cls, file_list_path: str = "data/lists/test.txt", **kwargs):
        """Default configuration for testing."""
        return cls(
            dataset=MIDIDatasetConfig(file_list_path=file_list_path),
            sampler=BucketSamplerConfig(
                batch_size=1,
                shuffle=False,
                drop_last=False
            ),
            num_workers=0,
            **kwargs
        )


class MIDIDataset(Dataset):
    """
    PyTorch Dataset for MIDI files using symusic.
    Converts MIDI to 3-channel piano roll representation: 
    Channel 0: Note On/Off (Binary)
    Channel 1: Velocity (Normalized 0-1)
    Channel 2: Tempo (Normalized)
    Shape: (3, T, 128) where T is time steps, 128 is pitch
    With optional data augmentation via random cropping.
    """
    
    def __init__(self, file_list_path: str, ticks_per_16th: int = 120,
                 random_crop: bool = True, crop_length: int = 256,
                 min_length: int = 64, augment_factor: int = 1,
                 pad_to_multiple: int = 4, cache_in_memory: bool = True,
                 cache_dir: Optional[str] = None,
                 pitch_shift_range: Tuple[int, int] = (-3, 3),
                 max_bar_len: int = 64):
        """
        Args:
            file_list_path: Path to txt file containing list of MIDI file paths
            ticks_per_16th: Number of ticks per 1/16 note after quantization
            random_crop: Enable random cropping for data augmentation
            crop_length: Target length in 16th notes (if random_crop=True)
            min_length: Minimum length for a valid crop
            augment_factor: How many random crops per MIDI file (effective dataset size multiplier)
            pitch_shift_range: Inclusive semitone range for pitch augmentation (min, max)
            max_bar_len: Maximum length of a bar for embedding
        """
        self.ticks_per_16th = ticks_per_16th
        self.random_crop = random_crop
        self.crop_length = crop_length
        self.min_length = min_length
        self.augment_factor = augment_factor
        self.pad_to_multiple = pad_to_multiple
        self.cache_in_memory = cache_in_memory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pitch_shift_range = pitch_shift_range
        self.pitch_shifts = list(range(pitch_shift_range[0], pitch_shift_range[1] + 1))
        self.pitch_shift_factor = max(len(self.pitch_shifts), 1)
        self.pitch_shift_enabled = any(shift != 0 for shift in self.pitch_shifts)
        self.crop_variants = self.augment_factor if self.random_crop else 1
        self._items_per_file = self.pitch_shift_factor * self.crop_variants
        self.max_bar_len = max_bar_len
        self.proc_config = ProcessingConfig(max_bar_len=max_bar_len)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {} if cache_in_memory else None
        self._failed_cache_paths = set()
        
        # Load file list
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.file_paths)} MIDI files from {file_list_path}")
        print(f"Dataset Init: random_crop={self.random_crop}, crop_length={self.crop_length}, min_length={self.min_length}")
        if self.random_crop and self.augment_factor > 1:
            print(f"Data augmentation enabled: {self.augment_factor}x ({self.augment_factor * len(self.file_paths)} effective samples)")
            print(f"Crop length: {self.crop_length} steps, Min length: {self.min_length} steps")
        if self.pitch_shift_enabled:
            print(f"Pitch shift augmentation range: [{self.pitch_shifts[0]}, {self.pitch_shifts[-1]}] semitones")
        print(f"Effective samples per file: {self._items_per_file} (crop variants={self.crop_variants}, pitch shifts={self.pitch_shift_factor})")
        print(f"Total nominal samples: {len(self)}")
        
        # Store duration for each file for bucketing
        self.durations = []
        self._precompute_durations()
    
    def _precompute_durations(self):
        """Precompute duration of each MIDI file for bucketing."""
        print("Precomputing MIDI durations for bucketing...")
        for idx, file_path in enumerate(self.file_paths):
            data = self._get_or_create_roll(idx, file_path, store_in_memory=self.cache_in_memory)
            if data is None:
                self.durations.append(0)
            else:
                # data is (notes, tempo, density)
                # notes shape is (2, T, 128), so duration is dim 1
                self.durations.append(int(data[0].shape[1]))
                if not self.cache_in_memory:
                    # Explicitly release reference to reduce peak memory usage
                    data = None

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(self.file_paths)} files")

        if self.durations:
            print(f"Duration range: {min(self.durations)} to {max(self.durations)} steps")
    
    def __len__(self):
        return len(self.file_paths) * self._items_per_file

    def _cache_path(self, file_path: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(file_path.encode('utf-8')).hexdigest()[:16]
        # Use 'v2' suffix for 3-channel cache to differentiate from previous cache
        return self.cache_dir / f"{digest}_v2.npz"

    def _load_cached_roll(self, file_path: str) -> Optional[np.ndarray]:
        cache_path = self._cache_path(file_path)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            cached = np.load(cache_path)["roll"]
            # Ensure it's 3D (3, T, 128)
            if cached.ndim == 2:
                # Old cache format (128, T) - ignore and rebuild
                return None
            return cached
        except Exception as exc:
            if cache_path not in self._failed_cache_paths:
                print(f"Warning: Failed to load cache {cache_path}: {exc}")
                self._failed_cache_paths.add(cache_path)
            return None

    def _save_cached_roll(self, file_path: str, piano_roll: np.ndarray) -> None:
        cache_path = self._cache_path(file_path)
        if cache_path is None or cache_path in self._failed_cache_paths:
            return
        try:
            np.savez_compressed(cache_path, roll=piano_roll.astype(np.float32))
        except Exception as exc:
            print(f"Warning: Failed to write cache {cache_path}: {exc}")
            self._failed_cache_paths.add(cache_path)

    def _build_piano_roll(self, file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build piano roll components from MIDI file.
        Returns: 
            notes: (2, T, 128) - [Note On/Off, Velocity]
            tempo: (T,) - Normalized Tempo
            density: (T,) - Note Density
            bar_pos: (T,) - Bar Position
        """
        try:
            score = symusic.Score(file_path)
        except Exception as e:
            print(f"Error loading MIDI {file_path}: {e}")
            return None

        score = score.resample(self.ticks_per_16th, min_dur=1)
        
        max_end_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                track_end = max(note.time + note.duration for note in track.notes)
                max_end_time = max(max_end_time, track_end)
        
        duration_steps = (max_end_time + self.ticks_per_16th - 1) // self.ticks_per_16th
        duration_steps = max(int(duration_steps), 1)

        # Create channels (T, 128)
        note_layer = np.zeros((duration_steps, 128), dtype=np.float32)
        vel_layer = np.zeros((duration_steps, 128), dtype=np.float32)
        
        # Fill Note and Velocity
        for track in score.tracks:
            for note in track.notes:
                pitch = note.pitch
                velocity = note.velocity / 127.0
                start_step = note.time // self.ticks_per_16th
                note_duration_steps = max(1, note.duration // self.ticks_per_16th)
                end_step = min(start_step + note_duration_steps, duration_steps)
                
                if 0 <= pitch < 128 and start_step < duration_steps:
                    note_layer[start_step:end_step, pitch] = 1.0
                    # Use max to handle overlapping notes in same track (monophonic assumption per pitch per track usually, but here we merge)
                    vel_layer[start_step:end_step, pitch] = np.maximum(vel_layer[start_step:end_step, pitch], velocity)

        # Calculate Density (Notes per step normalized)
        # Simple count of active notes at each step
        active_notes_per_step = np.sum(note_layer, axis=1) # (T,)
        # Normalize density: e.g. 0 to 10 notes -> 0 to 1. Log scale might be better but linear for now.
        # Let's clip at 12 notes (pretty dense)
        density_layer = np.clip(active_notes_per_step / 12.0, 0.0, 1.0).astype(np.float32)

        # Fill Tempo
        tempo_layer = np.zeros((duration_steps,), dtype=np.float32)
        tempos = score.tempos
        if not tempos:
            # Default 120 BPM
            default_bpm = 120.0
            norm_bpm = np.clip((default_bpm - 40) / 160, 0.0, 1.0)
            tempo_layer[:] = norm_bpm
        else:
            # Sort tempos by time
            tempos.sort(key=lambda x: x.time)
            
            current_bpm = 120.0
            if tempos[0].time > 0:
                current_bpm = 120.0 # Default at start if first tempo event is later
            
            tempo_idx = 0
            for t in range(duration_steps):
                tick_time = t * self.ticks_per_16th
                
                # Update current BPM if we passed a tempo change
                while tempo_idx < len(tempos) and tempos[tempo_idx].time <= tick_time:
                    current_bpm = tempos[tempo_idx].qpm
                    tempo_idx += 1
                
                norm_bpm = np.clip((current_bpm - 40) / 160, 0.0, 1.0)
                tempo_layer[t] = norm_bpm

        # Calculate Bar Pos (Dynamic)
        bar_pos_layer = compute_bar_pos(score, self.ticks_per_16th, self.proc_config)
        # Ensure length matches
        if len(bar_pos_layer) != duration_steps:
             # Resize if needed (should not happen if compute_bar_pos uses same logic but just in case)
             if len(bar_pos_layer) < duration_steps:
                  bar_pos_layer = np.pad(bar_pos_layer, (0, duration_steps - len(bar_pos_layer)), mode='edge')
             else:
                  bar_pos_layer = bar_pos_layer[:duration_steps]

        # Stack Note/Vel to (2, T, 128)
        notes_roll = np.stack([note_layer, vel_layer], axis=0)
        return notes_roll, tempo_layer, density_layer, bar_pos_layer

    def _get_or_create_roll(self, file_idx: int, file_path: str, store_in_memory: bool) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        roll = None
        if self.cache_in_memory and self._memory_cache is not None:
            roll = self._memory_cache.get(file_idx)

        if roll is None:
            # For now, disabling disk cache read for the new format to avoid complexity with existing cache files
            # Ideally we would check version or catch error, but rebuilding is safer for this refactor
            # cache_path = self._cache_path(file_path) ...
            
            try:
                roll = self._build_piano_roll(file_path)
            except Exception as exc:
                print(f"Warning: Failed to load {file_path}: {exc}")
                return None
            if roll is not None:
                # TODO: update save_cached_roll if needed, skipping for now
                if self.cache_in_memory and store_in_memory:
                    self._memory_cache[file_idx] = roll

        return roll

    def _get_piano_roll(self, file_idx: int, file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        data = self._get_or_create_roll(file_idx, file_path, store_in_memory=self.cache_in_memory)
        if data is None:
            return None
        # Return copies
        return (data[0].copy(), data[1].copy(), data[2].copy(), data[3].copy())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            notes: (2, T, 128)
            tempo: (T,)
            density: (T,)
            bar_pos: (T,)
            duration: int
            shift: int
        """
        base_len = len(self.file_paths)
        crop_variants = self.crop_variants
        pitch_variants = self.pitch_shift_factor
        items_per_file = crop_variants * pitch_variants

        file_idx = idx // items_per_file
        if file_idx >= base_len:
            file_idx = base_len - 1

        intra_idx = idx % items_per_file
        pitch_variant_idx = intra_idx // crop_variants if crop_variants > 0 else 0
        crop_variant_idx = intra_idx % crop_variants if crop_variants > 0 else 0
            
        file_path = self.file_paths[file_idx]
        
        try:
            data = self._get_piano_roll(file_idx, file_path)
            if data is None:
                raise RuntimeError("Failed to construct piano roll")

            notes, tempo, density, bar_pos = data
            
            # Shape notes: (2, T, 128), tempo: (T,), density: (T,), bar_pos: (T,)
            duration_steps = notes.shape[1]
            
            # Apply random cropping if enabled
            start_idx = 0
            if self.random_crop and duration_steps >= self.min_length:
                if duration_steps > self.crop_length:
                    # Random crop to crop_length
                    max_start = duration_steps - self.crop_length
                    if self.augment_factor > 1:
                        if max_start == 0:
                            start_idx = 0
                        else:
                            start_positions = np.linspace(0, max_start, num=self.augment_factor)
                            start_idx = int(np.round(start_positions[crop_variant_idx]))
                    else:
                        start_idx = np.random.randint(0, max_start + 1)
                    
                    # Crop along time dimension
                    notes = notes[:, start_idx:start_idx + self.crop_length, :]
                    tempo = tempo[start_idx:start_idx + self.crop_length]
                    density = density[start_idx:start_idx + self.crop_length]
                    bar_pos = bar_pos[start_idx:start_idx + self.crop_length]
                    duration_steps = self.crop_length
                # If duration_steps <= crop_length, keep full sequence
            elif self.crop_length is not None and self.crop_length > 0:
                # Even if random_crop is False, enforce crop_length if specified (deterministically take the start)
                # This ensures consistent visualization size (e.g. 512) for validation
                if duration_steps > self.crop_length:
                    notes = notes[:, :self.crop_length, :]
                    tempo = tempo[:self.crop_length]
                    density = density[:self.crop_length]
                    bar_pos = bar_pos[:self.crop_length]
                    duration_steps = self.crop_length
                elif duration_steps < self.crop_length:
                    # Pad to crop_length if shorter
                    if idx == 0 and not self.random_crop:
                        print(f"DEBUG: Padding validation sample {idx} from {duration_steps} to {self.crop_length}")
                    
                    pad_len = self.crop_length - duration_steps
                    notes_pad = np.zeros((2, pad_len, 128), dtype=notes.dtype)
                    notes = np.concatenate([notes, notes_pad], axis=1)
                    
                    tempo_pad = np.repeat(tempo[-1:], pad_len) if duration_steps > 0 else np.zeros(pad_len, dtype=tempo.dtype)
                    tempo = np.concatenate([tempo, tempo_pad], axis=0)
                    
                    density_pad = np.zeros(pad_len, dtype=density.dtype)
                    density = np.concatenate([density, density_pad], axis=0)
                    
                    # Continue bar pos sequence or pad 0
                    if duration_steps > 0:
                        last_val = bar_pos[-1]
                        pad_pos_seq = np.arange(1, pad_len + 1, dtype=bar_pos.dtype)
                        pad_bar_pos_vals = (last_val + pad_pos_seq) % 16
                        bar_pos = np.concatenate([bar_pos, pad_bar_pos_vals], axis=0)
                    else:
                        bar_pos = np.zeros(pad_len, dtype=bar_pos.dtype)
                    
                    duration_steps = self.crop_length
            
            # Apply pitch shift augmentation
            shift = self.pitch_shifts[pitch_variant_idx] if self.pitch_shifts else 0
            if shift != 0:
                notes = self._apply_pitch_shift(notes, shift)

            # Pad to ensure length is multiple of pad_to_multiple
            if self.pad_to_multiple > 1 and duration_steps > 0:
                padded_length = ((duration_steps + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
                if padded_length != duration_steps:
                    pad_width = padded_length - duration_steps
                    # Pad along dim 1 (time) for notes
                    notes = np.pad(notes, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
                    # Pad along dim 0 (time) for tempo/density/bar_pos
                    tempo = np.pad(tempo, (0, pad_width), mode='edge') # Pad tempo with edge value
                    density = np.pad(density, (0, pad_width), mode='constant') # Pad density with 0
                    
                    # Pad bar_pos continue the sequence - use modulo based on last seen value or naive continuation
                    # For complex meters, naive continuation might break if we cross a bar boundary and TS changes.
                    # However, since we don't have TS info here easily (lost in bar_pos), we'll try to infer cycle.
                    # Or safer: Pad with 0? Or Pad with Edge?
                    # Padding with continuation of the cycle logic is best if we assume constant TS at the end.
                    
                    # Heuristic: Look at the last few values to guess step.
                    last_val = bar_pos[-1]
                    second_last = bar_pos[-2] if len(bar_pos) > 1 else last_val - 1
                    step = last_val - second_last
                    if step <= 0: step = 1 # Fallback
                    
                    # We don't know the current bar length (TS), but we know the max is max_bar_len or the wrap point.
                    # Let's just assume the current bar length continues.
                    # If last_val is 15, next might be 0 or 16? 
                    # Let's just pad with 0s for safety or replicate edge.
                    # Replicating edge is bad for position.
                    # Let's extend assuming 4/4 (modulo 16) as a fallback for padding area which is usually ignored anyway.
                    pad_bar_pos = ((last_val + 1 + np.arange(pad_width)) % 16).astype(np.int64)
                    bar_pos = np.concatenate([bar_pos, pad_bar_pos])
                    
                    duration_steps = padded_length
            elif duration_steps == 0:
                duration_steps = self.pad_to_multiple
                notes = np.zeros((2, duration_steps, 128), dtype=np.float32)
                tempo = np.zeros((duration_steps,), dtype=np.float32)
                density = np.zeros((duration_steps,), dtype=np.float32)
                bar_pos = np.zeros((duration_steps,), dtype=np.int64)

            return torch.from_numpy(notes), torch.from_numpy(tempo), torch.from_numpy(density), torch.from_numpy(bar_pos), duration_steps, shift
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return empty on error
            fallback_length = self.crop_length if self.random_crop else max(self.min_length, self.pad_to_multiple)
            fallback_length = ((fallback_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
            if file_idx < len(self.durations):
                self.durations[file_idx] = fallback_length
            
            return (torch.zeros((2, fallback_length, 128), dtype=torch.float32),
                    torch.zeros((fallback_length,), dtype=torch.float32),
                    torch.zeros((fallback_length,), dtype=torch.float32),
                    torch.zeros((fallback_length,), dtype=torch.long),
                    fallback_length, 0)
    
    def get_duration(self, idx: int) -> int:
        """Get duration of a specific file."""
        return self.durations[idx]

    @staticmethod
    def _apply_pitch_shift(roll: np.ndarray, shift: int) -> np.ndarray:
        # roll shape: (3, T, 128)
        shifted = np.zeros_like(roll)
        # Shift along last dimension (pitch)
        if shift > 0:
            shifted[:, :, shift:] = roll[:, :, :-shift]
        elif shift < 0:
            shifted[:, :, :shift] = roll[:, :, -shift:]
        else:
            return roll.copy()
        return shifted


class BucketBatchSampler(Sampler):
    """
    Sampler that groups samples by duration into buckets for efficient batching.
    """
    
    def __init__(self, dataset: MIDIDataset, batch_size: int, 
                 bucket_boundaries: List[int] = None, drop_last: bool = False,
                 shuffle: bool = True, rank: int = 0, world_size: int = 1):
        """
        Args:
            dataset: MIDIDataset instance
            batch_size: Batch size
            bucket_boundaries: List of duration boundaries for bucketing
                              e.g., [100, 200, 300, 400, 500] creates 6 buckets
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples within each bucket
            rank: Process rank for DDP (default: 0)
            world_size: Total number of processes for DDP (default: 1)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        
        # Default bucket boundaries if not provided
        if bucket_boundaries is None:
            durations = dataset.durations
            max_dur = max(durations)
            # Create 10 buckets by default
            if max_dur > 0:
                bucket_boundaries = [int(max_dur * i / 10) for i in range(1, 10)]
            else:
                bucket_boundaries = [100]
        
        self.bucket_boundaries = sorted(bucket_boundaries)
        
        # Assign each sample to a bucket
        # Note: if dataset has augmentation, len(dataset) > len(dataset.durations)
        self.buckets = [[] for _ in range(len(self.bucket_boundaries) + 1)]
        
        if hasattr(dataset, 'random_crop') and dataset.random_crop:
            # Optimized path for augmented datasets
            # All augmented samples will have duration = crop_length (or original if shorter)
            print(f"Building buckets for augmented dataset ({len(dataset)} samples)...")
            
            # Assign all augmented samples
            crop_variants = getattr(dataset, 'crop_variants', dataset.augment_factor)
            pitch_shifts = getattr(dataset, 'pitch_shifts', [0])
            items_per_file = getattr(dataset, '_items_per_file', dataset.augment_factor)

            for file_idx in range(len(dataset.file_paths)):
                original_duration = dataset.durations[file_idx]
                
                if original_duration >= dataset.min_length:
                    effective_duration = min(dataset.crop_length, original_duration)
                else:
                    effective_duration = original_duration
                
                bucket_idx = self._get_bucket_idx(effective_duration)
                
                for pitch_idx, _ in enumerate(pitch_shifts):
                    for aug_idx in range(crop_variants):
                        sample_idx = (
                            file_idx * items_per_file
                            + pitch_idx * crop_variants
                            + aug_idx
                        )
                        if sample_idx < len(dataset):
                            self.buckets[bucket_idx].append(sample_idx)
            
            print(f"Bucket building completed.")
        else:
            # Original path for non-augmented datasets
            for idx in range(len(dataset)):
                duration = dataset.durations[idx]
                bucket_idx = self._get_bucket_idx(duration)
                self.buckets[bucket_idx].append(idx)
        
        # Print bucket statistics
        print(f"\nBucket statistics (batch_size={batch_size}):")
        total_samples = sum(len(b) for b in self.buckets)
        total_batches = sum(len(b) // batch_size for b in self.buckets)
        
        for i, bucket in enumerate(self.buckets):
            if i == 0:
                range_str = f"[0, {self.bucket_boundaries[0]})"
            elif i == len(self.buckets) - 1:
                range_str = f"[{self.bucket_boundaries[-1]}, ∞)"
            else:
                range_str = f"[{self.bucket_boundaries[i-1]}, {self.bucket_boundaries[i]})"
            
            num_batches = len(bucket) // batch_size
            print(f"  Bucket {i} {range_str}: {len(bucket)} samples, {num_batches} batches")
        
        print(f"Total: {total_samples} samples, {total_batches} batches")
    
    def _get_bucket_idx(self, duration: int) -> int:
        """Get bucket index for a given duration."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if duration < boundary:
                return i
        return len(self.bucket_boundaries)
    
    def __iter__(self):
        # Shuffle samples within each bucket
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(bucket)
        
        # Create batches from each bucket
        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Shuffle batches across buckets
        if self.shuffle:
            random.shuffle(batches)
        
        # For DDP: divide batches across ranks
        if self.world_size > 1:
            # Each rank gets a subset of batches
            batches_per_rank = len(batches) // self.world_size
            start_idx = self.rank * batches_per_rank
            if self.rank < self.world_size - 1:
                end_idx = start_idx + batches_per_rank
            else:
                # Last rank gets remaining batches
                end_idx = len(batches)
            batches = batches[start_idx:end_idx]
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        count = 0
        for bucket in self.buckets:
            if self.drop_last:
                count += len(bucket) // self.batch_size
            else:
                count += (len(bucket) + self.batch_size - 1) // self.batch_size
        
        # For DDP: return batches for this rank
        if self.world_size > 1:
            batches_per_rank = count // self.world_size
            if self.rank < self.world_size - 1:
                return batches_per_rank
            else:
                # Last rank gets remaining batches
                return count - batches_per_rank * (self.world_size - 1)
        
        return count


def collate_fn_pad(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]], 
                   patch_size: int = 32):
    """
    Collate function that pads components to the same length within a batch.
    
    Args:
        batch: List of (notes, tempo, density, bar_pos, duration, shift) tuples
        patch_size: Size of patches (default: 32).
    
    Returns:
        notes_batch: (B, 2, T, 128)
        tempo_batch: (B, T)
        density_batch: (B, T)
        bar_pos_batch: (B, T)
        durations: (B,)
        shifts: (B,)
    """
    # Unpack batch
    # batch item: (notes, tempo, density, bar_pos, duration, shift)
    if len(batch[0]) == 6:
        notes_list, tempo_list, density_list, bar_pos_list, durations, shifts = zip(*batch)
    elif len(batch[0]) == 5:
        # Backward compatibility or missing shift?
        # Check types to be safe, but usually we just added bar_pos
        # If user has old code calling this, it might break.
        # Let's assume it is (notes, tempo, density, bar_pos, duration)
        # Wait, __getitem__ returns 6 items now.
        notes_list, tempo_list, density_list, bar_pos_list, durations = zip(*batch)
        shifts = None
    else:
         # Fallback for safety
        notes_list, tempo_list, density_list, durations = zip(*batch)
        bar_pos_list = [torch.zeros_like(d) for d in density_list] # Dummy
        shifts = None

    max_duration = max(durations)
    padded_duration = ((max_duration + patch_size - 1) // patch_size) * patch_size
    
    padded_notes = []
    padded_tempos = []
    padded_densities = []
    padded_bar_pos = []
    
    for i in range(len(notes_list)):
        curr_notes = notes_list[i] # (2, T, 128)
        curr_tempo = tempo_list[i] # (T,)
        curr_density = density_list[i] # (T,)
        curr_bar_pos = bar_pos_list[i] # (T,)
        current_len = curr_notes.shape[1]
        
        pad_len = padded_duration - current_len
        
        if pad_len > 0:
            # Notes: pad right with 0
            notes_pad = torch.zeros((2, pad_len, 128), dtype=curr_notes.dtype)
            padded_notes.append(torch.cat([curr_notes, notes_pad], dim=1))
            
            # Tempo: pad right with edge value
            tempo_pad = curr_tempo[-1:].repeat(pad_len) if current_len > 0 else torch.zeros(pad_len, dtype=curr_tempo.dtype)
            padded_tempos.append(torch.cat([curr_tempo, tempo_pad], dim=0))
            
            # Density: pad right with 0
            density_pad = torch.zeros(pad_len, dtype=curr_density.dtype)
            padded_densities.append(torch.cat([curr_density, density_pad], dim=0))
            
            # Bar Pos: pad continuing the sequence
            last_pos = curr_bar_pos[-1].item() if current_len > 0 else -1
            pad_pos_seq = torch.arange(1, pad_len + 1, dtype=curr_bar_pos.dtype)
            pad_bar_pos_vals = (last_pos + pad_pos_seq) % 16
            padded_bar_pos.append(torch.cat([curr_bar_pos, pad_bar_pos_vals], dim=0))
            
        else:
            padded_notes.append(curr_notes)
            padded_tempos.append(curr_tempo)
            padded_densities.append(curr_density)
            padded_bar_pos.append(curr_bar_pos)
            
    notes_batch = torch.stack(padded_notes, dim=0)
    tempo_batch = torch.stack(padded_tempos, dim=0)
    density_batch = torch.stack(padded_densities, dim=0)
    bar_pos_batch = torch.stack(padded_bar_pos, dim=0)
    durations_tensor = torch.tensor(durations, dtype=torch.long)
    
    if shifts is not None:
        shifts_tensor = torch.tensor(shifts, dtype=torch.int)
        return notes_batch, tempo_batch, density_batch, bar_pos_batch, durations_tensor, shifts_tensor

    return notes_batch, tempo_batch, density_batch, bar_pos_batch, durations_tensor


def create_dataloader(file_list_path: str = None, batch_size: int = None, 
                      bucket_boundaries: List[int] = None,
                      shuffle: bool = None, num_workers: int = None,
                      drop_last: bool = None,
                      pin_memory: bool = True,
                      config: DataLoaderConfig = None) -> torch.utils.data.DataLoader:
    """Factory helper for constructing dataloaders with caching and bucketing support."""

    if config is None:
        assert file_list_path is not None, "file_list_path must be provided if config is None"
        if batch_size is None:
            batch_size = 1
        if shuffle is None:
            shuffle = True
        if num_workers is None:
            num_workers = 0
        if drop_last is None:
            drop_last = False

        dataset_cfg = MIDIDatasetConfig(file_list_path=file_list_path)
        sampler_cfg = BucketSamplerConfig(
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            shuffle=shuffle,
            drop_last=drop_last
        )

        config = DataLoaderConfig(
            dataset=dataset_cfg,
            sampler=sampler_cfg,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_bucket_sampler=not dataset_cfg.random_crop
        )

    dataset = MIDIDataset(
        file_list_path=config.dataset.file_list_path,
        ticks_per_16th=config.dataset.ticks_per_16th,
        random_crop=config.dataset.random_crop,
        crop_length=config.dataset.crop_length,
        min_length=config.dataset.min_length,
        augment_factor=config.dataset.augment_factor,
        pad_to_multiple=config.dataset.pad_to_multiple,
        cache_in_memory=config.dataset.cache_in_memory,
        cache_dir=config.dataset.cache_dir,
        pitch_shift_range=config.dataset.pitch_shift_range,
        max_bar_len=config.dataset.max_bar_len
    )

    collate_fn = functools.partial(collate_fn_pad, patch_size=config.patch_size)
    
    dataloader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "collate_fn": collate_fn,
    }
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor > 0:
            dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

    if not config.use_bucket_sampler:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.sampler.batch_size,
            shuffle=config.sampler.shuffle,
            drop_last=config.sampler.drop_last,
            **dataloader_kwargs,
        )
    else:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=config.sampler.batch_size,
            bucket_boundaries=config.sampler.bucket_boundaries,
            drop_last=config.sampler.drop_last,
            shuffle=config.sampler.shuffle,
            rank=rank,
            world_size=world_size
        )
        
        # Remove collate_fn from kwargs for clarity when using batch_sampler
        dataloader_kwargs_no_collate = dataloader_kwargs.copy()
        dataloader_kwargs_no_collate.pop("collate_fn", None)
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            **dataloader_kwargs_no_collate,
        )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    script_dir = Path(__file__).parent.parent.parent.absolute()
    train_file = script_dir / "data/lists" / "train.txt"
    
    print("Testing MIDIDataset...")
    print("=" * 60)
    
    # Create dataloader
    dataloader = create_dataloader(
        file_list_path=str(train_file),
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print("\n" + "=" * 60)
    print("Testing DataLoader...")
    print("=" * 60)
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        if len(batch) == 3:
            piano_rolls, durations, pitch_shifts = batch
        else:
            piano_rolls, durations = batch
            pitch_shifts = None
        print(f"\nBatch {i + 1}:")
        print(f"  Shape: {piano_rolls.shape}")
        print(f"  Durations: {durations.tolist()}")
        print(f"  Value range: [{piano_rolls.min():.3f}, {piano_rolls.max():.3f}]")
        print(f"  Non-zero ratio: {(piano_rolls != 0).sum().item() / piano_rolls.numel():.4f}")
        if pitch_shifts is not None:
            print(f"  Pitch shifts: {pitch_shifts.tolist()}")
        
        # Check channel stats
        print("  Channel stats:")
        print(f"    Note (Ch0): mean={piano_rolls[:, 0].mean():.4f}, max={piano_rolls[:, 0].max():.4f}")
        print(f"    Vel  (Ch1): mean={piano_rolls[:, 1].mean():.4f}, max={piano_rolls[:, 1].max():.4f}")
        print(f"    Tempo(Ch2): mean={piano_rolls[:, 2].mean():.4f}, min={piano_rolls[:, 2].min():.4f}, max={piano_rolls[:, 2].max():.4f}")
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\n" + "=" * 60)
    print("Dataset test completed!")
