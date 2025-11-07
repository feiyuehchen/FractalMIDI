import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union, Optional


# Create Logic Pro-style velocity colormap
# 0: black, low velocity: blue/cyan, mid velocity: green/yellow, high velocity: orange/red
def create_velocity_colormap():
    """
    Create a colormap similar to Logic Pro's velocity colors.
    Colors transition: Black -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red
    """
    colors = [
        (0.0, 0.0, 0.0),      # 0: Black (no note)
        (0.0, 0.0, 0.0),      # Very low: Still black (threshold)
        (0.0, 0.3, 0.8),      # Low: Blue
        (0.0, 0.7, 0.9),      # Low-mid: Cyan
        (0.0, 0.9, 0.5),      # Mid: Green
        (0.5, 1.0, 0.0),      # Mid-high: Yellow-green
        (1.0, 0.8, 0.0),      # High: Yellow-orange
        (1.0, 0.4, 0.0),      # Higher: Orange
        (1.0, 0.0, 0.0),      # Highest: Red
    ]
    
    # Create positions for each color (0 to 1)
    positions = [0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
    
    cmap = LinearSegmentedColormap.from_list('velocity', list(zip(positions, colors)))
    return cmap


# Global colormap instance
VELOCITY_CMAP = create_velocity_colormap()


def piano_roll_to_image(piano_roll: Union[torch.Tensor, np.ndarray],
                       apply_colormap: bool = True,
                       min_height: int = 256,
                       return_pil: bool = False) -> Union[torch.Tensor, Image.Image]:
    """
    Convert a piano roll matrix to an image.
    
    Args:
        piano_roll: Piano roll matrix of shape (128, T) or (batch, 128, T)
                   Values should be in range [-1, 1] or [0, 1] representing normalized velocity
                   -1 or 0 = silence (white), 1 = loud (colored)
        apply_colormap: If True, apply velocity colormap. If False, return grayscale
        min_height: Minimum height for the output image (will upscale if needed)
        return_pil: If True, return PIL Image. If False, return torch.Tensor
    
    Returns:
        Image tensor of shape (3, H, W) or (batch, 3, H, W) if input is batched,
        or PIL Image if return_pil=True
    """
    # Convert to numpy if tensor
    if isinstance(piano_roll, torch.Tensor):
        piano_roll_np = piano_roll.detach().cpu().numpy()
    else:
        piano_roll_np = piano_roll
    
    # Handle batched input
    is_batched = len(piano_roll_np.shape) == 3
    if not is_batched:
        piano_roll_np = piano_roll_np[np.newaxis, ...]  # Add batch dimension
    
    batch_size, height, width = piano_roll_np.shape
    
    # Normalize to [0, 1] range if input is in [-1, 1]
    # Check if values are in [-1, 1] range
    if piano_roll_np.min() < -0.5:  # Likely in [-1, 1] range
        # Convert from [-1, 1] to [0, 1]
        # -1 (silence/white) -> 0, 1 (loud) -> 1
        piano_roll_np = (piano_roll_np + 1.0) / 2.0
        piano_roll_np = np.clip(piano_roll_np, 0, 1)
    
    # Apply colormap to create RGB images
    if apply_colormap:
        # Apply colormap to each sample in batch
        rgb_images = []
        for i in range(batch_size):
            # Apply colormap (returns RGBA, we'll take RGB)
            colored = VELOCITY_CMAP(piano_roll_np[i])[:, :, :3]  # (H, W, 3)
            # Transpose to (3, H, W) for torch format
            colored = np.transpose(colored, (2, 0, 1))
            rgb_images.append(colored)
        rgb_images = np.stack(rgb_images, axis=0)  # (batch, 3, H, W)
    else:
        # Grayscale: replicate across 3 channels
        rgb_images = np.repeat(piano_roll_np[:, np.newaxis, :, :], 3, axis=1)
    
    # Convert to tensor
    result = torch.from_numpy(rgb_images).float()
    
    # Upscale if height is less than min_height
    if height < min_height:
        scale_factor = min_height // height
        if scale_factor > 1:
            result = torch.nn.functional.interpolate(
                result, 
                scale_factor=scale_factor, 
                mode="nearest"
            )
    
    # Remove batch dimension if input wasn't batched
    if not is_batched:
        result = result[0]
    
    # Convert to PIL if requested
    if return_pil:
        if is_batched:
            raise ValueError("Cannot return PIL Image for batched input")
        # Convert from (3, H, W) to (H, W, 3) and scale to [0, 255]
        img_np = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    return result


def visualize_piano_roll(piano_roll: Union[torch.Tensor, np.ndarray],
                        save_path: Optional[Union[str, Path]] = None,
                        title: Optional[str] = None,
                        figsize: tuple = (12, 4),
                        dpi: int = 100) -> Optional[Image.Image]:
    """
    Visualize a piano roll with proper labels and save to file.
    
    Args:
        piano_roll: Piano roll matrix of shape (128, T)
                   Values in range [-1, 1] or [0, 1]
        save_path: Path to save the visualization. If None, returns PIL Image
        title: Title for the plot
        figsize: Figure size in inches
        dpi: DPI for the saved image
    
    Returns:
        PIL Image if save_path is None, otherwise None
    """
    if isinstance(piano_roll, torch.Tensor):
        piano_roll_np = piano_roll.detach().cpu().numpy()
    else:
        piano_roll_np = piano_roll
    
    # Normalize to [0, 1] if in [-1, 1] range
    if piano_roll_np.min() < -0.5:
        piano_roll_np = (piano_roll_np + 1.0) / 2.0
        piano_roll_np = np.clip(piano_roll_np, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display with colormap
    im = ax.imshow(piano_roll_np, aspect='auto', origin='lower', 
                   cmap=VELOCITY_CMAP, vmin=0, vmax=1, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Velocity (normalized)')
    
    # Labels
    ax.set_xlabel('Time (1/16 notes)')
    ax.set_ylabel('Pitch')
    if title:
        ax.set_title(title)
    
    # Add pitch labels at octave boundaries
    octave_ticks = [12 * i for i in range(11)]  # C0, C1, ..., C10
    octave_labels = [f'C{i}' for i in range(11)]
    ax.set_yticks(octave_ticks)
    ax.set_yticklabels(octave_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None
    else:
        # Convert to PIL Image
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),
                             fig.canvas.tostring_rgb())
        plt.close(fig)
        return img


def visualize_batch(piano_rolls: torch.Tensor,
                   save_path: Optional[Union[str, Path]] = None,
                   nrow: int = 4,
                   padding: int = 2,
                   normalize: bool = False) -> Optional[torch.Tensor]:
    """
    Visualize a batch of piano rolls as a grid.
    Similar to torchvision.utils.save_image but for piano rolls with colormap.
    
    Args:
        piano_rolls: Batch of piano rolls, shape (batch, 128, T)
        save_path: Path to save the visualization
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize the values
    
    Returns:
        Image tensor if save_path is None, otherwise None
    """
    # Convert each piano roll to RGB image
    rgb_images = piano_roll_to_image(piano_rolls, apply_colormap=True)
    
    # Use torchvision's save_image
    if save_path:
        save_image(rgb_images, save_path, nrow=nrow, padding=padding, 
                  normalize=normalize, value_range=(0, 1))
        return None
    else:
        # Create grid without saving
        from torchvision.utils import make_grid
        grid = make_grid(rgb_images, nrow=nrow, padding=padding, 
                        normalize=normalize, value_range=(0, 1))
        return grid


def log_piano_roll_to_tensorboard(writer, tag: str, piano_roll: torch.Tensor,
                                  global_step: int, apply_colormap: bool = True):
    """
    Log a piano roll visualization to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the image
        piano_roll: Piano roll matrix of shape (128, T) or (batch, 128, T)
        global_step: Global step value
        apply_colormap: Whether to apply velocity colormap
    """
    # Convert to RGB image
    rgb_image = piano_roll_to_image(piano_roll, apply_colormap=apply_colormap)
    
    # Log to tensorboard
    if len(rgb_image.shape) == 3:
        # Single image
        writer.add_image(tag, rgb_image, global_step)
    else:
        # Batch of images
        writer.add_images(tag, rgb_image, global_step)


def compare_piano_rolls(original: Union[torch.Tensor, np.ndarray],
                       generated: Union[torch.Tensor, np.ndarray],
                       save_path: Optional[Union[str, Path]] = None,
                       figsize: tuple = (12, 8),
                       dpi: int = 100) -> Optional[Image.Image]:
    """
    Compare original and generated piano rolls side by side.
    
    Args:
        original: Original piano roll (128, T), values in [-1, 1] or [0, 1]
        generated: Generated piano roll (128, T), values in [-1, 1] or [0, 1]
        save_path: Path to save the comparison
        figsize: Figure size
        dpi: DPI for saved image
    
    Returns:
        PIL Image if save_path is None, otherwise None
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    # Normalize to [0, 1] if in [-1, 1] range
    if original.min() < -0.5:
        original = (original + 1.0) / 2.0
        original = np.clip(original, 0, 1)
    if generated.min() < -0.5:
        generated = (generated + 1.0) / 2.0
        generated = np.clip(generated, 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    
    # Original
    im1 = ax1.imshow(original, aspect='auto', origin='lower',
                    cmap=VELOCITY_CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Original')
    ax1.set_ylabel('Pitch')
    ax1.set_xlabel('Time (1/16 notes)')
    
    # Generated
    im2 = ax2.imshow(generated, aspect='auto', origin='lower',
                    cmap=VELOCITY_CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('Generated')
    ax2.set_ylabel('Pitch')
    ax2.set_xlabel('Time (1/16 notes)')
    
    # Shared colorbar
    fig.colorbar(im2, ax=[ax1, ax2], label='Velocity (normalized)', 
                orientation='vertical', pad=0.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),
                             fig.canvas.tostring_rgb())
        plt.close(fig)
        return img


if __name__ == "__main__":
    # Test the visualizer
    print("Testing MIDI Piano Roll Visualizer...")
    print("=" * 60)
    
    # Create a synthetic piano roll for testing
    print("\n1. Creating synthetic piano roll...")
    T = 128
    piano_roll = torch.zeros(128, T)
    
    # Add some notes with different velocities
    # C major chord (C4, E4, G4) with increasing velocity
    piano_roll[60, 0:16] = 0.5   # C4, medium velocity
    piano_roll[64, 0:16] = 0.7   # E4, higher velocity
    piano_roll[67, 0:16] = 0.9   # G4, very high velocity
    
    # Add a melody with varying velocities
    melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    for i, note in enumerate(melody_notes):
        start = 20 + i * 8
        velocity = 0.3 + (i / len(melody_notes)) * 0.6  # Crescendo
        piano_roll[note, start:start+8] = velocity
    
    # Add some bass notes
    piano_roll[48, 32:48] = 0.4   # C3, low velocity
    piano_roll[43, 48:64] = 0.6   # G2, medium velocity
    
    print(f"Piano roll shape: {piano_roll.shape}")
    print(f"Value range: [{piano_roll.min():.3f}, {piano_roll.max():.3f}]")
    print(f"Non-zero ratio: {(piano_roll != 0).sum().item() / piano_roll.numel():.4f}")
    
    # Test 1: Convert to image tensor
    print("\n2. Converting to RGB image tensor...")
    rgb_tensor = piano_roll_to_image(piano_roll, apply_colormap=True)
    print(f"RGB tensor shape: {rgb_tensor.shape}")
    
    # Test 2: Save visualization with labels
    print("\n3. Saving detailed visualization...")
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    visualize_piano_roll(
        piano_roll,
        save_path=output_dir / "test_piano_roll.png",
        title="Test Piano Roll"
    )
    print(f"Saved to: {output_dir / 'test_piano_roll.png'}")
    
    # Test 3: Batch visualization
    print("\n4. Testing batch visualization...")
    batch = torch.stack([piano_roll, piano_roll * 0.8, piano_roll * 0.6, piano_roll * 0.4])
    visualize_batch(
        batch,
        save_path=output_dir / "test_batch.png",
        nrow=2
    )
    print(f"Saved batch to: {output_dir / 'test_batch.png'}")
    
    # Test 4: Load a real MIDI file and visualize
    print("\n5. Testing with real MIDI file...")
    from dataset import MIDIDataset
    
    script_dir = Path(__file__).parent.absolute()
    train_file = script_dir / "dataset" / "train.txt"
    
    if train_file.exists():
        dataset = MIDIDataset(str(train_file))
        real_piano_roll, duration = dataset[0]
        
        # Crop to first 256 steps for visualization
        display_length = min(256, duration)
        cropped = real_piano_roll[:, :display_length]
        
        print(f"Real piano roll shape: {cropped.shape}")
        print(f"Value range: [{cropped.min():.3f}, {cropped.max():.3f}]")
        
        visualize_piano_roll(
            cropped,
            save_path=output_dir / "real_midi_example.png",
            title=f"Real MIDI Example (first {display_length} steps)"
        )
        print(f"Saved real MIDI to: {output_dir / 'real_midi_example.png'}")
    else:
        print("Train file not found, skipping real MIDI test")
    
    print("\n" + "=" * 60)
    print("Visualizer test completed!")
    print(f"All outputs saved to: {output_dir}")

