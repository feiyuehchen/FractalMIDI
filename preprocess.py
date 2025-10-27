import os
import random
from pathlib import Path

def generate_dataset_splits():
    """
    Generate train/valid/test splits for POP909 dataset.
    Each folder contains one MIDI file with the same name as the folder.
    Split ratio: 90% train, 5% valid, 5% test
    """
    # Get current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Base path for POP909 dataset
    base_path = Path.home() / "dataset" / "POP909-Dataset" / "POP909"
    
    # Collect all MIDI file paths
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
    
    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(midi_files)
    
    # Calculate split sizes
    total = len(midi_files)
    train_size = int(total * 0.90)
    valid_size = int(total * 0.05)
    # test_size will be the remainder to ensure all files are included
    
    # Split the data
    train_files = midi_files[:train_size]
    valid_files = midi_files[train_size:train_size + valid_size]
    test_files = midi_files[train_size + valid_size:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/total*100:.1f}%)")
    print(f"  Valid: {len(valid_files)} files ({len(valid_files)/total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} files ({len(test_files)/total*100:.1f}%)")
    
    # Save splits to text files in dataset subdirectory
    output_dir = script_dir / "dataset"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_files) + "\n")
    print(f"\nSaved train.txt")
    
    with open(output_dir / "valid.txt", "w") as f:
        f.write("\n".join(valid_files) + "\n")
    print(f"Saved valid.txt")
    
    with open(output_dir / "test.txt", "w") as f:
        f.write("\n".join(test_files) + "\n")
    print(f"Saved test.txt")
    
    return train_files, valid_files, test_files

if __name__ == "__main__":
    generate_dataset_splits()

