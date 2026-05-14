import argparse
import os
import torch
from ultralytics import YOLO

def check_hardware_device(requested_device):
    """Checks for hardware acceleration and returns the optimal device string."""
    if requested_device is not None and requested_device.lower() != "auto":
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Check: Found {torch.cuda.device_count()} GPU(s) -> [{gpu_name}]. Using CUDA.")
        return "0" 
    elif torch.backends.mps.is_available():
        print("Hardware Check: Found Apple Silicon. Using MPS.")
        return "mps"
    else:
        print("Hardware Check: No GPU detected. Falling back to CPU.")
        return "cpu"

