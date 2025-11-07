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
        file_list_path="dataset/train.txt"
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
    def training_default(cls, file_list_path: str = "dataset/train.txt", **kwargs):
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
    def validation_default(cls, file_list_path: str = "dataset/valid.txt", **kwargs):
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
    def test_default(cls, file_list_path: str = "dataset/test.txt", **kwargs):
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
    Converts MIDI to piano roll representation: 128 (pitch) x T (time steps)
    With optional data augmentation via random cropping.
    """
    
    def __init__(self, file_list_path: str, ticks_per_16th: int = 120,
                 random_crop: bool = True, crop_length: int = 256,
                 min_length: int = 64, augment_factor: int = 1,
                 pad_to_multiple: int = 4, cache_in_memory: bool = True,
                 cache_dir: Optional[str] = None,
                 pitch_shift_range: Tuple[int, int] = (-3, 3)):
        """
        Args:
            file_list_path: Path to txt file containing list of MIDI file paths
            ticks_per_16th: Number of ticks per 1/16 note after quantization
            random_crop: Enable random cropping for data augmentation
            crop_length: Target length in 16th notes (if random_crop=True)
            min_length: Minimum length for a valid crop
            augment_factor: How many random crops per MIDI file (effective dataset size multiplier)
            pitch_shift_range: Inclusive semitone range for pitch augmentation (min, max)
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
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {} if cache_in_memory else None
        self._failed_cache_paths = set()
        
        # Load file list
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.file_paths)} MIDI files from {file_list_path}")
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
            roll = self._get_or_create_roll(idx, file_path, store_in_memory=self.cache_in_memory)
            if roll is None:
                self.durations.append(0)
            else:
                self.durations.append(int(roll.shape[1]))
                if not self.cache_in_memory:
                    # Explicitly release reference to reduce peak memory usage
                    roll = None

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
        return self.cache_dir / f"{digest}.npz"

    def _load_cached_roll(self, file_path: str) -> Optional[np.ndarray]:
        cache_path = self._cache_path(file_path)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            cached = np.load(cache_path)["roll"]
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

    def _build_piano_roll(self, file_path: str) -> Optional[np.ndarray]:
        score = symusic.Score(file_path)
        score = score.resample(self.ticks_per_16th, min_dur=1)
        
        max_end_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                track_end = max(note.time + note.duration for note in track.notes)
                max_end_time = max(max_end_time, track_end)
        
        duration_steps = (max_end_time + self.ticks_per_16th - 1) // self.ticks_per_16th
        duration_steps = max(int(duration_steps), 1)

        piano_roll = np.zeros((128, duration_steps), dtype=np.float32)
        for track in score.tracks:
            for note in track.notes:
                pitch = note.pitch
                velocity = note.velocity / 127.0
                start_step = note.time // self.ticks_per_16th
                note_duration_steps = max(1, note.duration // self.ticks_per_16th)
                end_step = min(start_step + note_duration_steps, duration_steps)
                if 0 <= pitch < 128 and start_step < duration_steps:
                    piano_roll[pitch, start_step:end_step] = velocity

        return piano_roll

    def _get_or_create_roll(self, file_idx: int, file_path: str, store_in_memory: bool) -> Optional[np.ndarray]:
        roll = None
        if self.cache_in_memory and self._memory_cache is not None:
            roll = self._memory_cache.get(file_idx)

        if roll is None:
            roll = self._load_cached_roll(file_path)
            if roll is not None and self.cache_in_memory and store_in_memory:
                self._memory_cache[file_idx] = roll

        if roll is None:
            try:
                roll = self._build_piano_roll(file_path)
            except Exception as exc:
                print(f"Warning: Failed to load {file_path}: {exc}")
                return None
            if roll is not None:
                self._save_cached_roll(file_path, roll)
                if self.cache_in_memory and store_in_memory:
                    self._memory_cache[file_idx] = roll

        return roll

    def _get_piano_roll(self, file_idx: int, file_path: str) -> Optional[np.ndarray]:
        roll = self._get_or_create_roll(file_idx, file_path, store_in_memory=self.cache_in_memory)
        if roll is None:
            return None
        return roll.copy()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            piano_roll: Tensor of shape (128, T) where T is duration in 1/16 notes
            duration: Duration of the returned piano roll (after cropping if enabled)
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
            piano_roll = self._get_piano_roll(file_idx, file_path)
            if piano_roll is None:
                raise RuntimeError("Failed to construct piano roll")

            duration_steps = piano_roll.shape[1]
            
            # Apply random cropping if enabled
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
                    piano_roll = piano_roll[:, start_idx:start_idx + self.crop_length]
                    duration_steps = self.crop_length
                # If duration_steps <= crop_length, keep full sequence
            
            # Apply pitch shift augmentation
            shift = self.pitch_shifts[pitch_variant_idx] if self.pitch_shifts else 0
            if shift != 0:
                piano_roll = self._apply_pitch_shift(piano_roll, shift)

            # Pad to ensure length is multiple of pad_to_multiple
            if self.pad_to_multiple > 1 and duration_steps > 0:
                padded_length = ((duration_steps + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
                if padded_length != duration_steps:
                    pad_width = padded_length - duration_steps
                    piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant')
                    duration_steps = padded_length
            elif duration_steps == 0:
                duration_steps = self.pad_to_multiple
                piano_roll = np.zeros((128, duration_steps), dtype=np.float32)

            return torch.from_numpy(piano_roll), duration_steps, shift
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return empty piano roll on error
            fallback_length = self.crop_length if self.random_crop else max(self.min_length, self.pad_to_multiple)
            fallback_length = ((fallback_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
            if file_idx < len(self.durations):
                self.durations[file_idx] = fallback_length
            return torch.zeros((128, fallback_length), dtype=torch.float32), fallback_length, 0
    
    def get_duration(self, idx: int) -> int:
        """Get duration of a specific file."""
        return self.durations[idx]

    @staticmethod
    def _apply_pitch_shift(roll: np.ndarray, shift: int) -> np.ndarray:
        shifted = np.zeros_like(roll)
        if shift > 0:
            shifted[shift:, :] = roll[:-shift, :]
        elif shift < 0:
            shifted[:shift, :] = roll[-shift:, :]
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
            bucket_boundaries = [int(max_dur * i / 10) for i in range(1, 10)]
        
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


def collate_fn_pad(batch: List[Tuple[torch.Tensor, int]], 
                   patch_size: int = 32):
    """
    Collate function that pads piano rolls to the same length within a batch.
    The final length will be divisible by patch_size for proper patchification.
    
    Args:
        batch: List of (piano_roll, duration) tuples
        patch_size: Size of patches (default: 32). Final duration will be multiple of this.
    
    Returns:
        piano_rolls: Padded tensor of shape (batch_size, 128, padded_duration)
                    where padded_duration is divisible by patch_size
        durations: Tensor of original durations (batch_size,)
    """
    has_metadata = len(batch[0]) >= 3
    if has_metadata:
        piano_rolls, durations, metadata = zip(*batch)
    else:
        piano_rolls, durations = zip(*batch)
        metadata = None
    
    # Get maximum duration in this batch
    max_duration = max(durations)
    
    # Round up to nearest multiple of patch_size
    # This ensures the piano roll can be properly divided into patches
    padded_duration = ((max_duration + patch_size - 1) // patch_size) * patch_size
    
    # Pad all piano rolls to padded_duration
    padded_rolls = []
    for piano_roll in piano_rolls:
        _, current_duration = piano_roll.shape
        if current_duration < padded_duration:
            # Pad with zeros on the right
            padding = torch.zeros((128, padded_duration - current_duration), dtype=piano_roll.dtype)
            padded_roll = torch.cat([piano_roll, padding], dim=1)
        else:
            padded_roll = piano_roll
        padded_rolls.append(padded_roll)
    
    # Stack into batch
    piano_rolls_batch = torch.stack(padded_rolls, dim=0)
    durations_tensor = torch.tensor(durations, dtype=torch.long)
    
    if metadata is not None:
        metadata_tensor = torch.tensor(metadata, dtype=torch.int)
        return piano_rolls_batch, durations_tensor, metadata_tensor

    return piano_rolls_batch, durations_tensor


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
        pitch_shift_range=config.dataset.pitch_shift_range
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
    script_dir = Path(__file__).parent.absolute()
    train_file = script_dir / "dataset" / "train.txt"
    
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
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\n" + "=" * 60)
    print("Dataset test completed!")
