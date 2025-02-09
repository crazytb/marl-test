# src/utils/device_utils.py
import torch

def get_device(device_name=None):
    """
    기기 선택 함수
    Args:
        device_name: 'cuda', 'mps', 'cpu' 중 하나. None이면 자동 선택
    """
    if device_name is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device_name = device_name.lower()
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            print(f"Warning: {device_name} is not available. Using CPU instead.")
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
    return device