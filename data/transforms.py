"""Image transforms utilities."""
import torch
import torchvision.transforms.functional as TF

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (tensor + 1) / 2

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]"""
    return tensor * 2 - 1

def tensor_to_pil(tensor: torch.Tensor):
    """Tensor [C, H, W] in [-1, 1] -> PIL Image"""
    tensor = denormalize(tensor).clamp(0, 1)
    return TF.to_pil_image(tensor)