"""
Reconstruction Losses for message/image recovery.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for text steganography.
    Uses BCEWithLogitsLoss for bit-level reconstruction.
    """
    def __init__(self, loss_type: str = "bce", pos_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type.lower()

        if self.loss_type == "bce":
            # pos_weight > 1 increases penalty for false negatives (missed 1s)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        elif self.loss_type == "focal":
            self.loss_fn = None  # Custom implementation
        elif self.loss_type == "dice":
            self.loss_fn = None  # Custom implementation
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, msg_len] raw logits from decoder
            targets: [B, msg_len] binary targets (0/1)
        Returns:
            Scalar loss
        """
        if self.loss_type == "bce":
            return self.loss_fn(logits, targets.float())
        elif self.loss_type == "focal":
            return self._focal_loss(logits, targets)
        elif self.loss_type == "dice":
            return self._dice_loss(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for imbalanced bit classification."""
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * ce_loss).mean()

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for bit reconstruction."""
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + smooth) / (union + smooth)
        return (1 - dice).mean()


class ImageReconstructionLoss(nn.Module):
    """
    Reconstruction loss for image steganography.
    Combines L1 loss and optional LPIPS for secret image recovery.
    """
    def __init__(self, use_lpips: bool = True, lpips_weight: float = 1.0):
        super().__init__()
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        self.l1_loss = nn.L1Loss()

        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
            except ImportError:
                self.lpips_fn = None
                self.use_lpips = False

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            reconstructed: [B, 3, H, W] revealed secret
            target: [B, 3, H, W] original secret
        Returns:
            (total_loss, l1_loss, lpips_loss)
        """
        l1 = self.l1_loss(reconstructed, target)

        if self.use_lpips and self.lpips_fn is not None:
            lpips_val = self.lpips_fn(reconstructed.clamp(-1, 1), target.clamp(-1, 1)).mean()
            total = l1 + self.lpips_weight * lpips_val
            return total, l1, lpips_val
        else:
            return l1, l1, torch.tensor(0.0, device=reconstructed.device)


class CoverReconstructionLoss(nn.Module):
    """
    Reconstruction loss for cover image (stego vs cover).
    Simple MSE/L1 to preserve cover quality.
    """
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, stego: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(stego, cover)


def create_reconstruction_loss(config: dict, mode: str = "text") -> nn.Module:
    """Factory function."""
    if mode == "text":
        return ReconstructionLoss(loss_type="bce")
    else:
        return ImageReconstructionLoss(use_lpips=True, lpips_weight=1.0)