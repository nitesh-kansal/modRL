import torch
import torch.nn.functional as F
def get_device():
    if torch.cuda.is_available():
        device_name = "cuda:0"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device_name = "cpu"
    return torch.device(device_name) 