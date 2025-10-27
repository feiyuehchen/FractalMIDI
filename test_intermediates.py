import torch
from model import fractalmar_piano

print("Testing intermediate outputs capture...")

model = fractalmar_piano()
model.eval()

with torch.no_grad():
    # Test with return_intermediates=True
    result = model.sample(
        batch_size=1,
        num_iter_list=[8, 4, 2, 1],
        return_intermediates=True
    )
    
    if isinstance(result, tuple):
        final_output, intermediates = result
        print(f"\nFinal output shape: {final_output.shape}")
        print(f"Number of intermediates: {len(intermediates)}")
        for i, inter in enumerate(intermediates):
            print(f"  Level {inter['level']}: img_size={inter['img_size']}, shape={inter['output'].shape}")
    else:
        print(f"\nResult shape: {result.shape}")
        print("No intermediates returned (return_intermediates may not be working)")

print("\nTest completed!")
