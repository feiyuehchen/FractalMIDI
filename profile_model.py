
import torch
import torch.nn as nn
import time
from src.models.model_config import FractalModelConfig
from src.models.fractal_gen import RecursiveFractalNetwork
from src.training.trainer import FractalTrainerConfig

def profile_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on {device}")
    
    # Setup Config
    config = FractalTrainerConfig()
    config.model.architecture.seq_len_list = [32, 64, 128, 512]
    config.model.architecture.embed_dim_list = [256, 256, 128, 128]
    config.model.architecture.num_blocks_list = [4, 4, 2, 1]
    config.model.architecture.num_heads_list = [4, 4, 2, 1]
    config.model.generator.generator_type_list = ['mar', 'mar', 'mar', 'mar']
    config.model.generator.pitch_generator_type = 'parallel'
    
    model = RecursiveFractalNetwork(config.model).to(device)
    model.train()
    
    # Fake Batch
    B = 4
    T = 512
    notes = torch.randn(B, 2, T, 128).to(device)
    notes[:, 0] = (notes[:, 0] > 0).float() # Binary note on/off
    notes[:, 1] = torch.sigmoid(notes[:, 1]) # Velocity 0-1
    
    tempo = torch.rand(B, T).to(device)
    density = torch.rand(B, T).to(device)
    global_cond = torch.randn(B, T, 12).to(device)
    bar_pos = torch.randint(0, 16, (B, T)).to(device)
    
    print("\n--- Warmup ---")
    for _ in range(5):
        loss, _ = model(notes, tempo, density, global_cond, bar_pos)
        loss.backward()
        
    print("\n--- Profiling Forward+Backward ---")
    torch.cuda.synchronize()
    start = time.time()
    steps = 20
    for i in range(steps):
        t0 = time.time()
        loss, _ = model(notes, tempo, density, global_cond, bar_pos)
        torch.cuda.synchronize()
        t_fwd = time.time() - t0
        
        t1 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t1
        
        print(f"Step {i}: Fwd={t_fwd*1000:.1f}ms, Bwd={t_bwd*1000:.1f}ms, Total={(t_fwd+t_bwd)*1000:.1f}ms")
        
    avg_time = (time.time() - start) / steps
    print(f"\nAvg Total Time: {avg_time*1000:.2f} ms")
    print(f"Est. It/s: {1.0/avg_time:.2f}")

if __name__ == "__main__":
    profile_model()




