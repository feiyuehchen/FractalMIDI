
import torch
import time
from src.dataset.dataset import MIDIDataset
from torch.utils.data import DataLoader, default_collate
from src.dataset.dataset import collate_fn_pad
import functools
import os

def test_dataloader_throughput():
    print("\n--- DataLoader Throughput Test ---")
    
    # Config matching current training
    batch_size = 4
    num_workers = 8
    cache = False
    
    if os.path.exists("dataset/pop909/train.txt"):
        print(f"Config: Batch={batch_size}, Workers={num_workers}, Cache={cache}")
        
        dataset = MIDIDataset(
            file_list_path="dataset/pop909/train.txt",
            crop_length=512,
            augment_factor=10,  # Match training config
            cache_in_memory=cache
        )
        
        collate_fn = functools.partial(collate_fn_pad, patch_size=4)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            drop_last=True,
            collate_fn=collate_fn # Must use collate_fn_pad!
        )
        
        print("Warming up workers...")
        iter_loader = iter(dataloader)
        for _ in range(10):
            next(iter_loader)
            
        print("Measuring throughput (50 batches)...")
        start = time.time()
        for i in range(50):
            batch = next(iter_loader)
            # Simulate GPU transfer
            if torch.cuda.is_available():
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        item = item.to('cuda', non_blocking=True)
                        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        total_time = end - start
        avg_batch_time = total_time / 50
        it_per_sec = 50 / total_time
        
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg Batch Time: {avg_batch_time*1000:.2f} ms")
        print(f"Throughput: {it_per_sec:.2f} it/s")
        
        # Compare with Profiler Goal
        print(f"\nGoal (Model limit): ~16 it/s")
        print(f"Current Data limit: {it_per_sec:.2f} it/s")
        
        if it_per_sec < 10:
            print("\nWARNING: Data loading is significantly slower than model computation!")
        else:
            print("\nData loading seems fast enough.")
            
    else:
        print("dataset/pop909/train.txt not found")

if __name__ == "__main__":
    test_dataloader_throughput()
