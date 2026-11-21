"""
Checkpoint saving and loading utilities.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    max_keep: int = 5
):
    """
    Save training checkpoint.
    Args:
        state: Dictionary containing model, optimizer, epoch, metrics, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        max_keep: Maximum number of checkpoints to keep (by epoch)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), "best.pth")
        # Save directly instead of copy to avoid Windows file locking
        torch.save(state, best_path)

    # Clean old checkpoints (keep latest N)
    _cleanup_checkpoints(os.path.dirname(filepath), max_keep)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    Returns: Dictionary with epoch, metrics, etc.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Legacy format
        model.load_state_dict(checkpoint, strict=strict)

    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def _cleanup_checkpoints(checkpoint_dir: str, max_keep: int):
    """Remove old checkpoints, keeping only the latest max_keep."""
    checkpoints = list(Path(checkpoint_dir).glob("epoch_*.pth"))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))

    while len(checkpoints) > max_keep:
        oldest = checkpoints.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass


class CheckpointManager:
    """
    Manages checkpoint saving during training.
    Handles best model tracking, periodic saving, and cleanup.
    """
    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        save_best: bool = True,
        best_metric: str = "ber",
        best_mode: str = "min"  # "min" for BER/loss, "max" for PSNR/accuracy
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_value = float('inf') if best_mode == "min" else float('-inf')
        self.epoch = 0

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict] = None
    ):
        """Save checkpoint for current epoch."""
        self.epoch = epoch

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state:
            state.update(extra_state)

        # Save periodic checkpoint
        filepath = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
        save_checkpoint(state, str(filepath), is_best=False, max_keep=self.max_keep)

        # Check if best
        if self.save_best and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
            is_better = (
                (self.best_mode == "min" and current_value < self.best_value) or
                (self.best_mode == "max" and current_value > self.best_value)
            )

            if is_better:
                self.best_value = current_value
                best_filepath = self.checkpoint_dir / "best.pth"
                save_checkpoint(state, str(best_filepath), is_best=True, max_keep=self.max_keep)
                print(f"  [OK] New best model saved (epoch {epoch}, {self.best_metric}: {current_value:.6f})")

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best.pth"
        if best_path.exists():
            return load_checkpoint(str(best_path), model, optimizer, scheduler, device)
        else:
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load latest checkpoint by epoch number."""
        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        latest_path = checkpoints[-1]
        return load_checkpoint(str(latest_path), model, optimizer, scheduler, device)

    def get_best_value(self) -> float:
        return self.best_value