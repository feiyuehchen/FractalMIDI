
import torch
import time
from src.dataset.dataset import MIDIDataset
from torch.utils.data import DataLoader
import numpy as np

def test_speed():
    print("\n--- Dataset Loading Speed Test ---")
    try:
        import os
        if os.path.exists("dataset/pop909/train.txt"):
            # Test 1: No Cache
            print("Initializing Dataset (cache_in_memory=False)...")
            dataset = MIDIDataset(
                file_list_path="dataset/pop909/train.txt",
                crop_length=512,
                augment_factor=1, 
                cache_in_memory=False
            )
            
            print("Testing __getitem__ latency (disk I/O)...")
            start = time.time()
            for i in range(100):
                _ = dataset[i % len(dataset)]
            avg_time = (time.time() - start) / 100
            print(f"Avg __getitem__ time (No Cache): {avg_time*1000:.2f} ms")
            
            # Test 2: With Cache (Simulated single process)
            print("\nInitializing Dataset (cache_in_memory=True)...")
            dataset_cached = MIDIDataset(
                file_list_path="dataset/pop909/train.txt",
                crop_length=512,
                augment_factor=1, 
                cache_in_memory=True
            )
            
            # Populate cache
            print("Populating cache (first 100 items)...")
            start = time.time()
            for i in range(100):
                _ = dataset_cached[i]
            print(f"Cache population took {time.time()-start:.2f}s")
            
            print("Testing __getitem__ latency (Cached)...")
            start = time.time()
            for i in range(100):
                _ = dataset_cached[i]
            avg_time = (time.time() - start) / 100
            print(f"Avg __getitem__ time (Cached): {avg_time*1000:.2f} ms")
            
        else:
            print("dataset/pop909/train.txt not found")
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speed()




