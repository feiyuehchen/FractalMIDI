import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
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

# Create Tempo colormap (Blue -> Red)
def create_tempo_colormap():
    """
    Colormap for Tempo: Cool (Slow) to Hot (Fast).
    """
    colors = [
        (0.0, 0.0, 1.0),      # Slow: Blue
        (0.0, 1.0, 1.0),      # Med-Slow: Cyan
        (1.0, 1.0, 0.0),      # Med-Fast: Yellow
        (1.0, 0.0, 0.0),      # Fast: Red
    ]
    cmap = LinearSegmentedColormap.from_list('tempo', colors)
    return cmap

# Global colormap instances
VELOCITY_CMAP = create_velocity_colormap()
TEMPO_CMAP = create_tempo_colormap()


# ==============================================================================
# Easing Functions for Smooth Animations
# ==============================================================================

def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

def ease_out_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2


def piano_roll_to_image(piano_roll: Union[torch.Tensor, np.ndarray],
                       apply_colormap: bool = True,
                       min_height: int = 256,
                       return_pil: bool = False,
                       upscale_method: str = "nearest",
                       composite_tempo: bool = True) -> Union[torch.Tensor, Image.Image]:
    """
    Convert a piano roll matrix to an image with high quality.
    Supports both standard (128, T) and 3-channel (3, T, 128) formats.
    
    Args:
        piano_roll: Piano roll matrix.
                   Shape (128, T) or (B, 128, T): Old format.
                   Shape (3, T, 128) or (B, 3, T, 128): New format (Note, Vel, Tempo).
        apply_colormap: If True, apply velocity colormap. If False, return grayscale.
        min_height: Minimum height for the output image (will upscale if needed).
        return_pil: If True, return PIL Image. If False, return torch.Tensor.
        upscale_method: Interpolation method for upscaling.
        composite_tempo: If True and input is 3-channel, append tempo strip at bottom.
    
    Returns:
        Image tensor of shape (3, H, W) or (B, 3, H, W) if input is batched,
        or PIL Image if return_pil=True
    """
    # Convert to numpy if tensor
    if isinstance(piano_roll, torch.Tensor):
        piano_roll_np = piano_roll.detach().cpu().numpy()
    else:
        piano_roll_np = piano_roll
    
    # Detect input format
    ndim = piano_roll_np.ndim
    shape = piano_roll_np.shape
    
    is_batched = False
    is_3channel = False
    
    if ndim == 4:  # (B, 3, T, 128)
        is_batched = True
        is_3channel = True
        batch_size, channels, time_steps, pitch_bins = shape
        if channels != 3 or pitch_bins != 128:
             # Maybe (B, C, H, W) legacy? assume not for now based on plan
             pass
    elif ndim == 3:
        if shape[0] == 3 and shape[2] == 128: # (3, T, 128)
            is_3channel = True
            piano_roll_np = piano_roll_np[np.newaxis, ...] # Add batch
            is_batched = False # Treat as single item logically, remove batch at end
            batch_size = 1
        elif shape[1] == 128: # (B, 128, T) - Legacy
            is_batched = True
            batch_size, height, width = shape
        else: # (C, H, W) legacy single item or something else
             # Assume legacy (128, T) unbatched
             piano_roll_np = piano_roll_np[np.newaxis, ...]
             is_batched = False
             batch_size = 1
    elif ndim == 2: # (128, T) Legacy
        piano_roll_np = piano_roll_np[np.newaxis, ...]
        is_batched = False
        batch_size = 1
        
    # Process batch
    rgb_images = []
    
    for i in range(batch_size):
        item = piano_roll_np[i]
        
        if is_3channel:
            # Item shape: (3, T, 128)
            # Ch0: Note (0/1), Ch1: Vel (0-1), Ch2: Tempo (0-1)
            note_layer = item[0]
            vel_layer = item[1]
            tempo_layer = item[2]
            
            # Combine Note and Vel: Effective Vel = Vel * Note
            # Transpose to (128, T) for visualization
            effective_vel = (vel_layer * note_layer).T  # (128, T)
            
            # Tempo strip: average tempo over pitch (it should be broadcasted anyway)
            tempo_curve = tempo_layer.mean(axis=1)  # (T,)
            
            # 1. Render Piano Roll
            # Normalize/Clip
            effective_vel = np.clip(effective_vel, 0.0, 1.0)
            
            if apply_colormap:
                # (128, T, 3)
                roll_rgb = VELOCITY_CMAP(effective_vel)[:, :, :3]
                
                # Mask handling (legacy support for -1 mask, though new format uses 0/1)
                # If we want to visualize mask in new format, maybe Ch0 has special value?
                # Assuming standard 0-1 for now.
            else:
                roll_rgb = np.repeat(effective_vel[:, :, np.newaxis], 3, axis=2)
            
            # 2. Render Tempo Strip if requested
            if composite_tempo:
                tempo_height = max(16, effective_vel.shape[0] // 8)
                tempo_strip = np.tile(tempo_curve, (tempo_height, 1)) # (H_t, T)
                tempo_rgb = TEMPO_CMAP(tempo_strip)[:, :, :3] # (H_t, T, 3)
                
                # Add a separator line (black)
                separator = np.zeros((2, effective_vel.shape[1], 3))
                
                # Stack vertically: Piano Roll, Separator, Tempo
                # Note: Origin is usually lower-left for plots, but image arrays are top-down.
                # Matplotlib imshow(origin='lower') flips it.
                # Here we are building image array directly.
                # Piano roll (Pitch 0 at bottom) means index 0 should be top pitch?
                # In MIDI index 0 is usually low pitch.
                # If we want Pitch 0 at bottom, we need to flip dimension 0 of piano roll.
                roll_rgb = np.flipud(roll_rgb)
                
                full_rgb = np.vstack([roll_rgb, separator, tempo_rgb])
            else:
                full_rgb = np.flipud(roll_rgb)
            
            # Transpose to (3, H, W) for Torch
            full_rgb = np.transpose(full_rgb, (2, 0, 1))
            rgb_images.append(full_rgb)
            
        else:
            # Legacy (128, T)
            # ... (existing logic)
            height, width = item.shape
            
            # Normalize
            mask_indices = item < -0.01
            if item.min() < -0.5:
                item = np.maximum(item, 0.0)
                
            if apply_colormap:
                colored = VELOCITY_CMAP(item)[:, :, :3]
                
                # Handle mask
                if mask_indices.any():
                    void_color = np.array([0.2, 0.2, 0.2])
                    colored[mask_indices] = void_color
                
                # Flip UD to have low pitch at bottom
                # (Standard piano roll visualization)
                colored = np.flipud(colored)
                
                colored = np.transpose(colored, (2, 0, 1))
                rgb_images.append(colored)
            else:
                colored = np.repeat(item[:, :, np.newaxis], 3, axis=2)
                colored = np.flipud(colored)
                colored = np.transpose(colored, (2, 0, 1))
                rgb_images.append(colored)
                
    rgb_images = np.stack(rgb_images, axis=0) # (B, 3, H, W)
    
    # Convert to tensor
    result = torch.from_numpy(rgb_images).float()
    
    # Upscaling logic (same as before)
    _, channels, h, w = result.shape
    if h < min_height:
        scale_factor = min_height // h
        if scale_factor > 1:
            if upscale_method == "nearest":
                result = torch.nn.functional.interpolate(result, scale_factor=scale_factor, mode="nearest")
            elif upscale_method == "bilinear":
                result = torch.nn.functional.interpolate(result, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            # Lanczos omitted for brevity/torch compatibility
            
    # Remove batch dimension if input wasn't batched
    if not is_batched:
        result = result[0]
        
    # Return PIL
    if return_pil:
        if is_batched:
             raise ValueError("Cannot return PIL Image for batched input")
        img_np = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
        
    return result


def visualize_piano_roll(piano_roll: Union[torch.Tensor, np.ndarray],
                        save_path: Optional[Union[str, Path]] = None,
                        title: Optional[str] = None,
                        figsize: tuple = (12, 6),
                        dpi: int = 300) -> Optional[Image.Image]:
    """
    Visualize a piano roll with proper labels and save to file.
    Supports 3-channel input with Tempo strip.
    """
    # Convert to image tensor first using our robust converter
    # This handles 3-channel composition
    img_tensor = piano_roll_to_image(piano_roll, apply_colormap=True, min_height=256, composite_tempo=True)
    
    # Convert to numpy image (H, W, 3)
    if img_tensor.ndim == 4: img_tensor = img_tensor[0]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display
    # Note: img_np is already colored and flipped correctly by piano_roll_to_image
    im = ax.imshow(img_np, aspect='auto', interpolation='nearest')
    
    # Labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Pitch / Tempo')
    if title:
        ax.set_title(title)
    
    # Add simplified ticks
    ax.set_yticks([])
    ax.set_xticks([])
    
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


def visualize_batch(piano_rolls: torch.Tensor,
                   save_path: Optional[Union[str, Path]] = None,
                   nrow: int = 4,
                   padding: int = 2,
                   normalize: bool = False) -> Optional[torch.Tensor]:
    """
    Visualize a batch of piano rolls as a grid.
    """
    # Convert each piano roll to RGB image (handles 3-channel)
    rgb_images = piano_roll_to_image(piano_rolls, apply_colormap=True)
    
    # Use torchvision's save_image
    if save_path:
        save_image(rgb_images, save_path, nrow=nrow, padding=padding, 
                  normalize=normalize, value_range=(0, 1))
        return None
    else:
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
                            pop_effect: bool = False,
                            show_grid: bool = False,
                            show_progress: bool = False) -> Optional[List[Image.Image]]:
    """
    Create animation from intermediate steps.
    """
    if len(intermediates) == 0:
        return [] if save_path is None else None
    
    easing_functions = {
        "linear": lambda t: t,
        "ease_in_out_cubic": ease_in_out_cubic,
        "ease_out_back": ease_out_back,
        "ease_out_elastic": ease_out_elastic,
        "ease_in_out_sine": ease_in_out_sine,
    }
    easing_func = easing_functions.get(easing, lambda t: t)
    
    frames = []
    frames_per_transition = max(1, int(fps * transition_duration))
    
    for i in range(len(intermediates) - 1):
        current = intermediates[i]
        next_step = intermediates[i+1]
        
        if isinstance(current, dict) and 'output' in current: current = current['output']
        if isinstance(next_step, dict) and 'output' in next_step: next_step = next_step['output']
        
        if isinstance(current, torch.Tensor): current = current.detach().cpu().float()
        if isinstance(next_step, torch.Tensor): next_step = next_step.detach().cpu().float()
            
        # Handle 3-channel tensors (squeeze batch dim if present)
        if current.ndim == 4: current = current[0] # (3, T, 128)
        if next_step.ndim == 4: next_step = next_step[0]

        # Match dimensions (Time axis is dim 1)
        if current.shape[1] != next_step.shape[1]:
            # Upsample the smaller one to match the larger one
            if current.shape[1] < next_step.shape[1]:
                # (C, T, P) -> (1, C, T, P) -> Upsample -> (C, T_new, P)
                # We need to transpose to (N, C, P, T) for 2D interpolation or just 1D on T?
                # Piano roll is (3, T, 128). We want to stretch T.
                # F.interpolate takes (N, C, H, W). Let's treat T as H, 128 as W.
                current = current.unsqueeze(0) # (1, 3, T, 128)
                current = torch.nn.functional.interpolate(
                    current, 
                    size=(next_step.shape[1], next_step.shape[2]), 
                    mode='nearest'
                )
                current = current.squeeze(0)
            else:
                # Downsample current (rare, but possible if visualizing reverse?)
                next_step = next_step.unsqueeze(0)
                next_step = torch.nn.functional.interpolate(
                    next_step, 
                    size=(current.shape[1], current.shape[2]), 
                    mode='nearest'
                )
                next_step = next_step.squeeze(0)

        # Detect diff
        diff = (next_step - current).clamp(min=0) if pop_effect else None
        
        for f in range(frames_per_transition):
            t = f / frames_per_transition
            alpha = easing_func(t)
            
            interpolated = current * (1 - alpha) + next_step * alpha
            
            if pop_effect and f < frames_per_transition // 2 and diff is not None:
                pop_intensity = (1 - 2 * t) * 0.3
                interpolated = interpolated + diff * pop_intensity
                interpolated = torch.clamp(interpolated, min=-1.0, max=1.0)
            
            img = piano_roll_to_image(
                interpolated, 
                apply_colormap=True, 
                return_pil=True,
                min_height=min_height,
                upscale_method="nearest"
            )
            
            if show_grid or show_progress:
                total_transitions = len(intermediates) - 1
                current_progress = (i + t) / total_transitions if total_transitions > 0 else 1.0
                img = add_visual_effects(img, show_grid=show_grid, show_progress=show_progress, progress=current_progress)
            
            frames.append(img)
            
    # Final frame
    final = intermediates[-1]
    if isinstance(final, dict) and 'output' in final: final = final['output']
    if isinstance(final, torch.Tensor): final = final.detach().cpu().float()
    if final.ndim == 4: final = final[0]
    
    final_img = piano_roll_to_image(
        final, 
        apply_colormap=True, 
        return_pil=True,
        min_height=min_height,
        upscale_method="nearest"
    )
    
    final_hold_frames = int(fps * final_hold)
    for _ in range(final_hold_frames):
        frames.append(final_img)
        
    if save_path:
        if len(frames) > 0:
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
    from PIL import ImageDraw, ImageFont
    img = img.copy()
    if img.mode != 'RGBA': img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    
    if show_grid:
        actual_interval = max(20, width // 32)
        for x in range(0, width, actual_interval):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        octave_height = height // 12 # Roughly
        for y in range(0, height, octave_height):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
            
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    return img


def log_piano_roll_to_tensorboard(writer, tag: str, piano_roll: torch.Tensor,
                                  global_step: int, apply_colormap: bool = True):
    rgb_image = piano_roll_to_image(piano_roll, apply_colormap=apply_colormap)
    if len(rgb_image.shape) == 3:
        writer.add_image(tag, rgb_image, global_step)
    else:
        writer.add_images(tag, rgb_image, global_step)

def compare_piano_rolls(original, generated, save_path=None, figsize=(12, 10), dpi=100):
    """
    Compare original and generated piano rolls side by side.
    """
    # Use our improved visualization
    img_orig = piano_roll_to_image(original, apply_colormap=True, return_pil=True, composite_tempo=True)
    img_gen = piano_roll_to_image(generated, apply_colormap=True, return_pil=True, composite_tempo=True)
    
    # Create figure to display both
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    
    ax1.imshow(np.array(img_orig), aspect='auto', interpolation='nearest')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(np.array(img_gen), aspect='auto', interpolation='nearest')
    ax2.set_title('Generated')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        return img

if __name__ == "__main__":
    # Test the visualizer with 3-channel data
    print("Testing MIDI Piano Roll Visualizer (v2 3-channel)...")
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic 3-channel roll (3, T, 128)
    T = 256
    roll = torch.zeros(3, T, 128)
    
    # Ch0: Notes
    roll[0, 10:50, 60] = 1.0 # C4
    roll[0, 60:100, 64] = 1.0 # E4
    roll[0, 110:150, 67] = 1.0 # G4
    
    # Ch1: Velocity (Ramp up)
    for t in range(T):
        roll[1, t, :] = t / T
        
    # Ch2: Tempo (Slow to Fast)
    for t in range(T):
        # BPM ramping
        roll[2, t, :] = 0.2 + 0.8 * (t / T)
        
    print(f"3-channel roll shape: {roll.shape}")
    
    # Test 1: Save image
    print("Saving 3-channel visualization...")
    visualize_piano_roll(roll, save_path=output_dir / "test_v2_roll.png", title="3-Channel Test")
    print(f"Saved to {output_dir / 'test_v2_roll.png'}")
    
    # Test 2: Batch
    print("Testing batch...")
    batch = torch.stack([roll, roll, roll])
    visualize_batch(batch, save_path=output_dir / "test_v2_batch.png")
    print(f"Saved to {output_dir / 'test_v2_batch.png'}")
