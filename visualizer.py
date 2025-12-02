import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union, Optional, List
import io
import math


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


# ==============================================================================
# Easing Functions for Smooth Animations
# ==============================================================================

def ease_in_out_cubic(t: float) -> float:
    """
    Cubic ease-in-out easing function.
    Starts slow, speeds up in the middle, slows down at the end.
    
    Args:
        t: Progress value between 0 and 1
        
    Returns:
        Eased value between 0 and 1
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """
    Back ease-out easing function.
    Creates a "pop" effect by overshooting and settling back.
    
    Args:
        t: Progress value between 0 and 1
        overshoot: Amount of overshoot (default 1.70158)
        
    Returns:
        Eased value (may exceed 1 temporarily)
    """
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def ease_out_elastic(t: float) -> float:
    """
    Elastic ease-out easing function.
    Creates a bouncy "spring" effect.
    
    Args:
        t: Progress value between 0 and 1
        
    Returns:
        Eased value (may exceed 1 temporarily)
    """
    if t == 0 or t == 1:
        return t
    
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_in_out_sine(t: float) -> float:
    """
    Sine ease-in-out easing function.
    Smooth acceleration and deceleration.
    
    Args:
        t: Progress value between 0 and 1
        
    Returns:
        Eased value between 0 and 1
    """
    return -(math.cos(math.pi * t) - 1) / 2


def piano_roll_to_image(piano_roll: Union[torch.Tensor, np.ndarray],
                       apply_colormap: bool = True,
                       min_height: int = 256,
                       return_pil: bool = False,
                       upscale_method: str = "nearest") -> Union[torch.Tensor, Image.Image]:
    """
    Convert a piano roll matrix to an image with high quality.
    
    Args:
        piano_roll: Piano roll matrix of shape (128, T) or (batch, 128, T)
                   Values should be in range [-1, 1] or [0, 1] representing normalized velocity
                   -1 or 0 = silence (black), 1 = loud (colored)
        apply_colormap: If True, apply velocity colormap. If False, return grayscale
        min_height: Minimum height for the output image (will upscale if needed)
        return_pil: If True, return PIL Image. If False, return torch.Tensor
        upscale_method: Interpolation method for upscaling ("nearest", "bilinear", "lanczos")
    
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
    # But preserve -1 (mask) for separate handling
    # Create mask for negative values (typically -1)
    mask_indices = piano_roll_np < -0.01
    
    if piano_roll_np.min() < -0.5:  # Likely in [-1, 1] range
        # If we have negatives (like -1 for mask), clip them to 0 for colormap
        piano_roll_np = np.maximum(piano_roll_np, 0.0)
    
    # Apply colormap to create RGB images
    if apply_colormap:
        # Apply colormap to each sample in batch
        rgb_images = []
        for i in range(batch_size):
            # Apply colormap (returns RGBA, we'll take RGB)
            colored = VELOCITY_CMAP(piano_roll_np[i])[:, :, :3]  # (H, W, 3)
            
            # Apply white/gray color to masked areas
            # Use a dark gray color for "void" to reduce contrast flicker with black background
            void_color = np.array([0.2, 0.2, 0.2]) # Dark gray
            
            # Get mask for this item
            item_mask = mask_indices[i]
            if item_mask.any():
                # Broadcast void_color to match masked area
                colored[item_mask] = void_color
                
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
            # Use specified interpolation method
            if upscale_method == "nearest":
                result = torch.nn.functional.interpolate(
                    result, 
                    scale_factor=scale_factor, 
                    mode="nearest"
                )
            elif upscale_method == "bilinear":
                result = torch.nn.functional.interpolate(
                    result,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False
                )
            else:  # lanczos - use PIL for better quality
                result_pil = []
                for i in range(result.size(0)):
                    img_np = (result[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    new_size = (width * scale_factor, height * scale_factor)
                    pil_img = pil_img.resize(new_size, Image.LANCZOS)
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    result_pil.append(torch.from_numpy(img_np).permute(2, 0, 1))
                result = torch.stack(result_pil, dim=0)
    
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
                        dpi: int = 300) -> Optional[Image.Image]:
    """
    Visualize a piano roll with proper labels and save to file.
    
    Args:
        piano_roll: Piano roll matrix of shape (128, T)
                   Values in range [-1, 1] or [0, 1]
        save_path: Path to save the visualization. If None, returns PIL Image
        title: Title for the plot
        figsize: Figure size in inches
        dpi: DPI for the saved image
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


def create_growth_animation(intermediates: List[torch.Tensor],
                            save_path: Optional[Union[str, Path]] = None,
                            fps: int = 30,
                            transition_duration: float = 0.1,
                            final_hold: float = 1.0,
                            min_height: int = 512,
                            optimize: bool = True,
                            quality: int = 95,
                            easing: str = "ease_in_out_cubic",
                            pop_effect: bool = False,  # Disabled by default to prevent flickering
                            show_grid: bool = False,
                            show_progress: bool = False) -> Optional[List[Image.Image]]:
    """
    Create a high-quality smooth growth animation with note-popping effects.
    Interpolates between steps with easing functions for natural motion.
    
    Args:
        intermediates: List of tensors (H, W) or (1, H, W) representing generation steps.
        save_path: Path to save the GIF.
        fps: Frames per second.
        transition_duration: Duration of transition between steps in seconds.
        final_hold: How long to hold the final frame in seconds.
        min_height: Minimum height for output images (higher = better quality).
        optimize: Whether to optimize GIF for smaller file size.
        quality: Quality setting for optimization (1-100, higher = better).
        easing: Easing function to use ("linear", "ease_in_out_cubic", "ease_out_back", 
                "ease_out_elastic", "ease_in_out_sine").
        pop_effect: Whether to add a subtle brightness pop effect for new notes.
        show_grid: Whether to show grid overlay on frames.
        show_progress: Whether to show progress indicator on frames.
        
    Returns:
        List of PIL Images (frames) if save_path is None.
    """
    if len(intermediates) == 0:
        return [] if save_path is None else None
    
    # Select easing function
    easing_functions = {
        "linear": lambda t: t,
        "ease_in_out_cubic": ease_in_out_cubic,
        "ease_out_back": ease_out_back,
        "ease_out_elastic": ease_out_elastic,
        "ease_in_out_sine": ease_in_out_sine,
    }
    easing_func = easing_functions.get(easing, lambda t: t)
    
    frames = []
    
    # Number of interpolated frames between steps
    frames_per_transition = max(1, int(fps * transition_duration))
    
    for i in range(len(intermediates) - 1):
        current = intermediates[i]
        next_step = intermediates[i+1]
        
        # Extract tensor from dict if needed
        if isinstance(current, dict) and 'output' in current:
            current = current['output']
        if isinstance(next_step, dict) and 'output' in next_step:
            next_step = next_step['output']
        
        # Ensure tensors are on CPU and float
        if isinstance(current, torch.Tensor):
            current = current.detach().cpu().float()
        if isinstance(next_step, torch.Tensor):
            next_step = next_step.detach().cpu().float()
            
        # Squeeze if needed
        if current.ndim == 4: current = current.squeeze(0).squeeze(0) # (1, 1, H, W) -> (H, W)
        elif current.ndim == 3: current = current.squeeze(0) # (1, H, W) -> (H, W)
        
        if next_step.ndim == 4: next_step = next_step.squeeze(0).squeeze(0)
        elif next_step.ndim == 3: next_step = next_step.squeeze(0)
        
        # Detect new notes (difference between steps)
        if pop_effect:
            diff = (next_step - current).clamp(min=0)  # Only new/increased notes
        
        # Interpolate with easing
        for f in range(frames_per_transition):
            t = f / frames_per_transition
            alpha = easing_func(t)
            
            # Interpolate between current and next
            interpolated = current * (1 - alpha) + next_step * alpha
            
            # Add pop effect for new notes
            if pop_effect and f < frames_per_transition // 2:
                # Add temporary brightness boost to new notes in first half of transition
                pop_intensity = (1 - 2 * t) * 0.3  # Fades from 0.3 to 0
                interpolated = interpolated + diff * pop_intensity
                # Use min=-1.0 to preserve mask values (-1) instead of clamping them to 0 (black)
                interpolated = torch.clamp(interpolated, min=-1.0, max=1.0)
            
            # Convert to high-quality image
            img = piano_roll_to_image(
                interpolated, 
                apply_colormap=True, 
                return_pil=True,
                min_height=min_height,
                upscale_method="nearest"  # Nearest for crisp piano roll
            )
            
            # Add visual effects
            if show_grid or show_progress:
                # Calculate overall progress
                total_transitions = len(intermediates) - 1
                current_progress = (i + t) / total_transitions if total_transitions > 0 else 1.0
                
                img = add_visual_effects(
                    img,
                    show_grid=show_grid,
                    show_progress=show_progress,
                    progress=current_progress
                )
            
            frames.append(img)
            
    # Add final frame
    final = intermediates[-1]
    if isinstance(final, dict) and 'output' in final:
        final = final['output']
        
    if isinstance(final, torch.Tensor):
        final = final.detach().cpu().float()
    
    if final.ndim == 4: final = final.squeeze(0).squeeze(0)
    elif final.ndim == 3: final = final.squeeze(0)
    
    final_img = piano_roll_to_image(
        final, 
        apply_colormap=True, 
        return_pil=True,
        min_height=min_height,
        upscale_method="nearest"
    )
    
    # Hold final frame
    final_hold_frames = int(fps * final_hold)
    for _ in range(final_hold_frames):
        frames.append(final_img)
        
    if save_path:
        # Save as high-quality GIF
        if len(frames) > 0:
            # Use optimize and quality settings for better compression
            frames[0].save(
                save_path,
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=1000 // fps,
                loop=0,
                optimize=optimize,
                quality=quality
            )
        return None
    
    return frames


def add_visual_effects(img: Image.Image, 
                       show_grid: bool = True,
                       show_progress: bool = False,
                       progress: float = 0.0,
                       grid_color: tuple = (80, 80, 80, 128),
                       grid_interval: int = 16) -> Image.Image:
    """
    Add visual effects to a piano roll image.
    
    Args:
        img: PIL Image to add effects to
        show_grid: Whether to show grid lines
        show_progress: Whether to show progress indicator
        progress: Progress value (0.0 to 1.0)
        grid_color: RGBA color for grid lines
        grid_interval: Interval between grid lines (in time steps)
        
    Returns:
        PIL Image with effects added
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a copy to avoid modifying original
    img = img.copy()
    
    # Convert to RGBA for transparency support
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create overlay for grid
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = img.size
    
    # Add grid lines
    if show_grid:
        # Vertical grid lines (time divisions)
        # Assuming width represents time, add lines every grid_interval pixels
        # Scale grid_interval based on image width
        actual_interval = max(20, width // 32)  # At least 20px, or width/32
        
        for x in range(0, width, actual_interval):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        
        # Horizontal grid lines (octave divisions)
        # 128 pitches, 12 pitches per octave = ~10.67 octaves
        octave_height = height // 11  # Approximate octave height
        
        for y in range(0, height, octave_height):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Add progress indicator
    # Disabled as per user request
    # if show_progress and 0.0 <= progress <= 1.0:
    if False:
        # Progress bar at the top
        bar_height = 4
        bar_width = int(width * progress)
        
        # Background bar (dark)
        draw.rectangle(
            [(0, 0), (width, bar_height)],
            fill=(40, 40, 40, 200)
        )
        
        # Progress bar (bright)
        if bar_width > 0:
            draw.rectangle(
                [(0, 0), (bar_width, bar_height)],
                fill=(0, 200, 150, 255)  # Cyan-green color
            )
        
        # Progress text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        progress_text = f"{progress*100:.0f}%"
        text_bbox = draw.textbbox((0, 0), progress_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text with background
        text_x = width - text_width - 10
        text_y = 10
        draw.rectangle(
            [(text_x - 5, text_y - 2), (text_x + text_width + 5, text_y + text_height + 2)],
            fill=(0, 0, 0, 180)
        )
        draw.text((text_x, text_y), progress_text, fill=(255, 255, 255, 255), font=font)
    
    # Composite overlay onto image
    img = Image.alpha_composite(img, overlay)
    
    # Convert back to RGB
    img = img.convert('RGB')
    
    return img


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
    
    # Test 4: Animation
    print("\n5. Testing growth animation...")
    intermediates = []
    # Create progressive growth
    steps = 10
    for i in range(steps + 1):
        ratio = i / steps
        # Interpolate between -1 (canvas init) and final piano_roll
        # -1 maps to 0 in visualization, so we can just use piano_roll * ratio
        # Actually, let's simulate filling
        current = piano_roll.clone()
        # Mask out parts to simulate AR generation
        mask_idx = int(T * ratio)
        current[:, mask_idx:] = -1
        intermediates.append(current)
        
    create_growth_animation(
        intermediates,
        save_path=output_dir / "test_growth.gif",
        fps=15,
        transition_duration=0.2
        )
    print(f"Saved animation to: {output_dir / 'test_growth.gif'}")
    
    print("\n" + "=" * 60)
    print("Visualizer test completed!")
    print(f"All outputs saved to: {output_dir}")
