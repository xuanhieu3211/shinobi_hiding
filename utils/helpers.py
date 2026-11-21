"""
Helper utilities for training.
"""
import torch
import random
import numpy as np
import time
from typing import Tuple, Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def move_to_device(batch, device: torch.device):
    """Recursively move batch to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def setup_amp(model: torch.nn.Module, optimizer: torch.optim.Optimizer, enabled: bool = True):
    """Setup Automatic Mixed Precision (AMP)."""
    if enabled and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        return scaler
    return None


def setup_channels_last(model: torch.nn.Module, enabled: bool = True):
    """Convert model to channels-last memory format."""
    if enabled and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model


def compile_model(model: torch.nn.Module, enabled: bool = True, mode: str = "reduce-overhead"):
    """Compile model with torch.compile (PyTorch 2.0+)."""
    if enabled and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=mode)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    return model


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    lr = config.get("lr", 1e-4)
    beta1 = config.get("beta1", 0.5)
    beta2 = config.get("beta2", 0.999)
    weight_decay = config.get("weight_decay", 1e-5)

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_epochs: int):
    """Create learning rate scheduler from config."""
    scheduler_type = config.get("scheduler", "cosine_warmup")
    warmup_epochs = config.get("warmup_epochs", 5)

    if scheduler_type == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    elif scheduler_type == "step":
        step_size = config.get("step_size", num_epochs // 3)
        gamma = config.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)


def save_images(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = "",
    nrow: int = 8,
    normalize: bool = True
):
    """Save batch of images to disk."""
    from torchvision.utils import save_image
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if normalize:
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]

    save_path = Path(save_dir) / f"{prefix}.png"
    save_image(images, str(save_path), nrow=nrow, normalize=False)
    return str(save_path)


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True
) -> torch.Tensor:
    """Create grid of images for visualization."""
    from torchvision.utils import make_grid as tv_make_grid

    if normalize:
        images = (images + 1) / 2

    return tv_make_grid(images, nrow=nrow, padding=padding, normalize=False)


class Timer:
    """Simple timer context manager."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"{self.name}: {format_time(self.elapsed)}")

    def reset(self):
        self.start_time = time.perf_counter()

    def get_elapsed(self) -> float:
        return time.perf_counter() - self.start_time