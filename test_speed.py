import torch
import time
from src.dataset.dataset import MIDIDataset, collate_fn_pad
from torch.utils.data import DataLoader
from src.models.components import FractalInputProj, ParallelOutputProj
from src.models.fractal_gen import FractalBlock, RecursiveFractalNetwork
from src.models.model_config import FractalModelConfig, ArchitectureConfig, GeneratorConfig, PianoRollConfig
from src.training.trainer import FractalMIDILightningModule, FractalTrainerConfig

def test_speed():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    # 模擬真實訓練的 Batch Size
    B = 32 
    print(f"Batch Size: {B}")

    # Config matching fractal_midi_v2_clean.yaml (Optimized)
    config = FractalModelConfig(
        architecture=ArchitectureConfig(
            seq_len_list=[32, 64, 128, 512],
            embed_dim_list=[256, 256, 128, 128],
            num_blocks_list=[4, 4, 2, 1],
            num_heads_list=[4, 4, 2, 1],
            input_channels=2,
            compressed_dim=64
        ),
        generator=GeneratorConfig(
            generator_type_list=['mar', 'mar', 'mar', 'mar'],
            # 關鍵：使用 Parallel
            pitch_generator_type='parallel'
        ),
        piano_roll=PianoRollConfig(
            num_velocity_bins=32
        )
    )
    
    # 1. Full Recursive Model Speed Test
    print("\n--- RecursiveFractalNetwork (Full Model, Parallel) ---")
    model = RecursiveFractalNetwork(config).to(device)
    
    # Enable Gradient Checkpointing manually if supported by model (or simulate overhead)
    # The current implementation might not utilize the config flag directly in the model init, 
    # but let's assume standard forward pass first.
    
    T = 512
    notes = torch.randn(B, 2, T, 128).to(device)
    tempo = torch.randn(B, T).to(device)
    density = torch.randn(B, T).to(device)
    global_cond = torch.randn(B, T, 12).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.cuda.amp.autocast(enabled=True):
         for _ in range(5): 
             loss, _ = model(notes, tempo, density, global_cond)
             loss.backward()
             model.zero_grad()
             
    print("Testing Forward+Backward speed...")
    start = time.time()
    num_steps = 20
    
    with torch.cuda.amp.autocast(enabled=True):
        for _ in range(num_steps):
            loss, _ = model(notes, tempo, density, global_cond)
            loss.backward()
            model.zero_grad()
            
    torch.cuda.synchronize()
    total_time = time.time() - start
    avg_time = total_time / num_steps
    print(f"Full Model Forward+Backward (B={B}): {avg_time*1000:.2f} ms")
    print(f"Throughput: {B / avg_time:.2f} samples/sec")

    # 2. Dataset Loading (Cached)
    print("\n--- Dataset Loading (Simulated Cached) ---")
    try:
        import os
        if os.path.exists("dataset/pop909/train.txt"):
            dataset = MIDIDataset(
                file_list_path="dataset/pop909/train.txt",
                crop_length=512,
                augment_factor=1, 
                cache_in_memory=True # Enable Cache
            )
            print(f"Dataset size: {len(dataset)}")
            
            # Populate cache first to be fair
            print("Populating cache (reading first 50 items)...")
            for i in range(50): _ = dataset[i]
            
            dataloader = DataLoader(
                dataset, 
                batch_size=B, 
                num_workers=2, # As per optimized config
                collate_fn=collate_fn_pad
            )
            
            print("Warming up dataloader...")
            iter_dl = iter(dataloader)
            next(iter_dl)
            
            start = time.time()
            for i in range(20):
                _ = next(iter_dl)
            avg_time = (time.time() - start) / 20
            print(f"DataLoader (num_workers=2, B={B}) Avg Time per Batch: {avg_time*1000:.2f} ms")
            
        else:
            print("dataset/pop909/train.txt not found")
    except Exception as e:
        print(f"Dataset test failed: {e}")

    # 3. Lightning Module Full Step Test (Forward + Backward + Optimizer)
    print("\n--- Lightning Module Training Step (End-to-End) ---")
    trainer_config = FractalTrainerConfig(
        model=config,
        max_steps=1000,
        precision="16"
    )
    
    pl_module = FractalMIDILightningModule(trainer_config).to(device)
    pl_module.train()
    
    # Create dummy batch matching collate_fn output structure
    # (notes, tempo, density, bar_pos, chroma, durations, shifts)
    dummy_batch = (
        notes,              # Reusing tensors from above
        tempo,
        density,
        torch.zeros(B, T, device=device).long(), # bar_pos
        global_cond,        # chroma
        torch.zeros(B, device=device), # durations
        torch.zeros(B, device=device)  # shifts
    )
    
    optimizer = torch.optim.AdamW(pl_module.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    print("Warming up Lightning Step...")
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=True):
            loss = pl_module.training_step(dummy_batch, 0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    print("Testing Lightning Step speed...")
    start = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=True):
            loss = pl_module.training_step(dummy_batch, 0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    total_time = time.time() - start
    avg_time = total_time / num_steps
    print(f"Lightning Module Step (B={B}): {avg_time*1000:.2f} ms")
    print(f"Throughput: {B / avg_time:.2f} samples/sec")

if __name__ == "__main__":
    test_speed()
