import torch
from torch.utils.data import Dataset, Sampler
import symusic
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import random


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
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.ticks_per_16th > 0, "ticks_per_16th must be positive"


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
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.patch_size > 0, "patch_size must be positive"
    
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
                 min_length: int = 64, augment_factor: int = 100):
        """
        Args:
            file_list_path: Path to txt file containing list of MIDI file paths
            ticks_per_16th: Number of ticks per 1/16 note after quantization
            random_crop: Enable random cropping for data augmentation
            crop_length: Target length in 16th notes (if random_crop=True)
            min_length: Minimum length for a valid crop
            augment_factor: How many random crops per MIDI file (effective dataset size multiplier)
        """
        self.ticks_per_16th = ticks_per_16th
        self.random_crop = random_crop
        self.crop_length = crop_length
        self.min_length = min_length
        self.augment_factor = augment_factor
        
        # Load file list
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.file_paths)} MIDI files from {file_list_path}")
        if self.random_crop:
            print(f"Data augmentation enabled: {self.augment_factor}x ({self.augment_factor * len(self.file_paths)} effective samples)")
            print(f"Crop length: {self.crop_length} steps, Min length: {self.min_length} steps")
        
        # Store duration for each file for bucketing
        self.durations = []
        self._precompute_durations()
    
    def _precompute_durations(self):
        """Precompute duration of each MIDI file for bucketing."""
        print("Precomputing MIDI durations for bucketing...")
        for idx, file_path in enumerate(self.file_paths):
            try:
                score = symusic.Score(file_path)
                # Quantize to 1/16 note
                score = score.resample(self.ticks_per_16th, min_dur=1)
                
                # Get maximum end time across all tracks
                max_end_time = 0
                for track in score.tracks:
                    if len(track.notes) > 0:
                        track_end = max(note.time + note.duration for note in track.notes)
                        max_end_time = max(max_end_time, track_end)
                
                # Convert to number of 1/16 note steps
                duration_steps = (max_end_time + self.ticks_per_16th - 1) // self.ticks_per_16th
                self.durations.append(duration_steps)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.file_paths)} files")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                self.durations.append(0)
        
        print(f"Duration range: {min(self.durations)} to {max(self.durations)} steps")
    
    def __len__(self):
        if self.random_crop:
            # Augmented dataset size: augment_factor * number of files
            # This makes the dataset appear 100x larger
            return len(self.file_paths) * self.augment_factor
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            piano_roll: Tensor of shape (128, T) where T is duration in 1/16 notes
            duration: Duration of the returned piano roll (after cropping if enabled)
        """
        # Map augmented idx to actual file idx
        if self.random_crop:
            file_idx = idx % len(self.file_paths)
        else:
            file_idx = idx
            
        file_path = self.file_paths[file_idx]
        
        try:
            # Load and quantize MIDI
            score = symusic.Score(file_path)
            score = score.resample(self.ticks_per_16th, min_dur=1)
            
            # Get duration
            duration_steps = self.durations[file_idx]
            
            # Create full piano roll: 128 (pitch) x T (time steps)
            piano_roll = np.zeros((128, duration_steps), dtype=np.float32)
            
            # Merge all tracks
            for track in score.tracks:
                for note in track.notes:
                    pitch = note.pitch
                    velocity = note.velocity / 127.0  # Normalize to [0, 1]
                    
                    # Calculate start and end time steps
                    start_step = note.time // self.ticks_per_16th
                    note_duration_steps = max(1, note.duration // self.ticks_per_16th)
                    end_step = min(start_step + note_duration_steps, duration_steps)
                    
                    # Fill the piano roll with velocity for the note duration
                    if 0 <= pitch < 128 and start_step < duration_steps:
                        piano_roll[pitch, start_step:end_step] = velocity
            
            # Apply random cropping if enabled
            if self.random_crop and duration_steps >= self.min_length:
                if duration_steps > self.crop_length:
                    # Random crop to crop_length
                    max_start = duration_steps - self.crop_length
                    start_idx = np.random.randint(0, max_start + 1)
                    piano_roll = piano_roll[:, start_idx:start_idx + self.crop_length]
                    duration_steps = self.crop_length
                # If duration_steps <= crop_length, keep full sequence
            
            return torch.from_numpy(piano_roll), duration_steps
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return empty piano roll on error
            fallback_length = self.crop_length if self.random_crop else 1
            return torch.zeros((128, fallback_length), dtype=torch.float32), fallback_length
    
    def get_duration(self, idx: int) -> int:
        """Get duration of a specific file."""
        return self.durations[idx]


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
        
        # Check if dataset has augmentation
        if hasattr(dataset, 'random_crop') and dataset.random_crop:
            # Optimized path for augmented datasets
            # All augmented samples will have duration = crop_length (or original if shorter)
            print(f"Building buckets for augmented dataset ({len(dataset)} samples)...")
            
            # Find which bucket crop_length belongs to
            crop_bucket_idx = self._get_bucket_idx(dataset.crop_length)
            
            # Assign all augmented samples
            # For each original file, create augment_factor samples
            for file_idx in range(len(dataset.file_paths)):
                original_duration = dataset.durations[file_idx]
                
                # Determine actual duration after cropping
                if original_duration >= dataset.min_length:
                    effective_duration = min(dataset.crop_length, original_duration)
                else:
                    effective_duration = original_duration
                
                bucket_idx = self._get_bucket_idx(effective_duration)
                
                # Add all augmented versions of this file to the bucket
                for aug_idx in range(dataset.augment_factor):
                    sample_idx = file_idx + aug_idx * len(dataset.file_paths)
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
                   patch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
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
    piano_rolls, durations = zip(*batch)
    
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
    
    return piano_rolls_batch, durations_tensor


def create_dataloader(file_list_path: str = None, batch_size: int = None, 
                      bucket_boundaries: List[int] = None,
                      shuffle: bool = None, num_workers: int = None,
                      drop_last: bool = None,
                      pin_memory: bool = True,
                      config: DataLoaderConfig = None) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with bucketing and padding.
    
    Args:
        file_list_path: Path to txt file with MIDI file paths (deprecated, use config)
        batch_size: Batch size (deprecated, use config)
        bucket_boundaries: Duration boundaries for bucketing (deprecated, use config)
        shuffle: Whether to shuffle data (deprecated, use config)
        num_workers: Number of worker processes (deprecated, use config)
        drop_last: Whether to drop last incomplete batch (deprecated, use config)
        config: DataLoaderConfig object (recommended)
    
    Returns:
        DataLoader instance
    
    Examples:
        # Using config (recommended)
        >>> config = DataLoaderConfig.training_default("dataset/train.txt")
        >>> dataloader = create_dataloader(config=config)
        
        # Using parameters (deprecated but still supported)
        >>> dataloader = create_dataloader("dataset/train.txt", batch_size=8)
    """
    # If config is provided, use it
    if config is not None:
        dataset = MIDIDataset(
            file_list_path=config.dataset.file_list_path,
            ticks_per_16th=config.dataset.ticks_per_16th,
            random_crop=config.dataset.random_crop,
            crop_length=config.dataset.crop_length,
            min_length=config.dataset.min_length,
            augment_factor=config.dataset.augment_factor
        )
        
        # Create collate function with patch_size
        from functools import partial
        collate_fn = partial(collate_fn_pad, patch_size=config.patch_size)
        
        # Decide whether to use bucket sampler
        # If using random_crop (all same length), use simple DataLoader
        if config.dataset.random_crop and not config.use_bucket_sampler:
            print(f"Using simple DataLoader (all samples cropped to {config.dataset.crop_length})")
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=config.sampler.batch_size,
                shuffle=config.sampler.shuffle,
                collate_fn=collate_fn,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=config.sampler.drop_last
            )
        else:
            # Use bucket sampler for variable-length sequences
            print(f"Using BucketBatchSampler for variable-length sequences")
            # Get DDP info if available
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
            
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
    else:
        # Legacy parameter-based interface
        assert file_list_path is not None, "file_list_path must be provided if config is None"
        assert batch_size is not None, "batch_size must be provided if config is None"
        
        # Set defaults for optional parameters
        if shuffle is None:
            shuffle = True
        if num_workers is None:
            num_workers = 4
        if drop_last is None:
            drop_last = False
        
        dataset = MIDIDataset(file_list_path)
        
        # Get DDP info if available
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            drop_last=drop_last,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size
        )
        
        # Use default patch_size=32 for legacy interface
        from functools import partial
        collate_fn = partial(collate_fn_pad, patch_size=32)
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
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
    for i, (piano_rolls, durations) in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        print(f"  Shape: {piano_rolls.shape}")
        print(f"  Durations: {durations.tolist()}")
        print(f"  Value range: [{piano_rolls.min():.3f}, {piano_rolls.max():.3f}]")
        print(f"  Non-zero ratio: {(piano_rolls != 0).sum().item() / piano_rolls.numel():.4f}")
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\n" + "=" * 60)
    print("Dataset test completed!")
