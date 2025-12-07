import os
import random
import argparse
from pathlib import Path

def collect_pop909_files(base_path):
    """
    Collect MIDI files from POP909 dataset.
    Each folder (001-909) contains one MIDI file with the same name as the folder.
    """
    midi_files = []
    missing_files = []
    
    # Iterate through folders 001 to 909
    for i in range(1, 910):
        folder_name = f"{i:03d}"
        midi_path = base_path / folder_name / f"{folder_name}.mid"
        
        # Check if file exists
        if midi_path.exists():
            midi_files.append(str(midi_path.absolute()))
        else:
            missing_files.append(str(midi_path))
    
    print(f"Found {len(midi_files)} MIDI files")
    if missing_files:
        print(f"Warning: {len(missing_files)} files are missing")
        for f in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    return midi_files

def collect_aria_midi_files(base_path):
    """
    Collect MIDI files from aria-midi-v1-unique-ext dataset.
    MIDI files are organized in multiple subdirectories (aa, ab, ..., de).
    """
    midi_files = []
    
    print(f"Scanning directory: {base_path}")
    
    # Recursively find all .mid files
    for midi_path in sorted(base_path.rglob("*.mid")):
        if midi_path.is_file():
            midi_files.append(str(midi_path.absolute()))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    return midi_files

def generate_dataset_splits(dataset_type, output_dir=None):
    """
    Generate train/valid/test splits for specified dataset.
    Split ratio: 99.8% train, 0.1% valid, 0.1% test
    
    Args:
        dataset_type: Either 'pop909' or 'ariamidi'
        output_dir: Optional custom output directory
    """
    # Get current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Determine base path based on dataset type
    if dataset_type == "pop909":
        base_path = Path.home() / "dataset" / "POP909-Dataset" / "POP909"
        midi_files = collect_pop909_files(base_path)
    elif dataset_type == "ariamidi":
        base_path = Path.home() / "dataset" / "aria-midi-v1-unique-ext" / "data"
        midi_files = collect_aria_midi_files(base_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'pop909' or 'ariamidi'")
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {base_path}")
    
    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(midi_files)
    
    # Calculate split sizes
    total = len(midi_files)
    valid_size = max(1, int(total * 0.001))  # 0.1% for validation, at least 1 file
    test_size = max(1, int(total * 0.001))   # 0.1% for test, at least 1 file
    train_size = total - valid_size - test_size  # remaining for train (~99.8%)
    
    # Split the data
    train_files = midi_files[:train_size]
    valid_files = midi_files[train_size:train_size + valid_size]
    test_files = midi_files[train_size + valid_size:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/total*100:.2f}%)")
    print(f"  Valid: {len(valid_files)} files ({len(valid_files)/total*100:.2f}%)")
    print(f"  Test:  {len(test_files)} files ({len(test_files)/total*100:.2f}%)")
    
    # Save splits to text files in dataset/{dataset_type} subdirectory
    if output_dir is None:
        output_dir = script_dir / "dataset" / dataset_type
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_files) + "\n")
    print(f"\nSaved {output_dir / 'train.txt'}")
    
    with open(output_dir / "valid.txt", "w") as f:
        f.write("\n".join(valid_files) + "\n")
    print(f"Saved {output_dir / 'valid.txt'}")
    
    with open(output_dir / "test.txt", "w") as f:
        f.write("\n".join(test_files) + "\n")
    print(f"Saved {output_dir / 'test.txt'}")
    
    return train_files, valid_files, test_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate train/valid/test splits for MIDI datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pop909", "ariamidi"],
        help="Dataset type: 'pop909' for POP909 dataset, 'ariamidi' for aria-midi-v1-unique-ext dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for split files (default: ./dataset)"
    )
    
    args = parser.parse_args()
    generate_dataset_splits(args.dataset, args.output_dir)

