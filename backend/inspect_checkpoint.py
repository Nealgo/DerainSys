import torch
import sys
import os

def inspect(path):
    try:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        state_dict = torch.load(path, map_location='cpu')
        print(f"Keys in {os.path.basename(path)}:")
        keys = list(state_dict.keys())
        for k in keys:
            print(f"  {k}: {state_dict[k].shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    inspect(sys.argv[1])
