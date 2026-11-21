"""
Metrics for steganography evaluation.
PSNR, SSIM, BER, Bit Accuracy, etc.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Calculate PSNR between two images.
    Args:
        img1, img2: [B, 3, H, W] or [3, H, W] tensors in range [-1, 1]
        data_range: Max - min of pixel values (2.0 for [-1, 1])
    Returns:
        PSNR in dB (mean over batch)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Convert to numpy
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    psnrs = []
    for i in range(img1_np.shape[0]):
        # Convert from CHW to HWC
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        psnr_val = skimage_psnr(im1, im2, data_range=data_range)
        psnrs.append(psnr_val)

    return float(np.mean(psnrs))


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Calculate SSIM between two images.
    Args:
        img1, img2: [B, 3, H, W] or [3, H, W] tensors in range [-1, 1]
        data_range: Max - min of pixel values
    Returns:
        SSIM (mean over batch)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    ssims = []
    for i in range(img1_np.shape[0]):
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        # channel_axis=2 for RGB
        ssim_val = skimage_ssim(im1, im2, data_range=data_range, channel_axis=2)
        ssims.append(ssim_val)

    return float(np.mean(ssims))


def calculate_ber(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor
) -> float:
    """
    Calculate Bit Error Rate (BER).
    Args:
        predicted_bits: [B, msg_len] predicted bits (0/1 or probabilities)
        target_bits: [B, msg_len] ground truth bits (0/1)
    Returns:
        BER as percentage (0-100)
    """
    # If probabilities, threshold at 0.5
    if predicted_bits.max() > 1.0 or predicted_bits.min() < 0.0:
        predicted_bits = (predicted_bits > 0.5).float()

    errors = (predicted_bits != target_bits).float().sum()
    total = target_bits.numel()
    ber = (errors / total * 100).item()
    return ber


def calculate_bit_accuracy(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor
) -> float:
    """
    Calculate Bit Accuracy.
    Args:
        predicted_bits: [B, msg_len] predicted bits (0/1 or probabilities)
        target_bits: [B, msg_len] ground truth bits (0/1)
    Returns:
        Accuracy as percentage (0-100)
    """
    if predicted_bits.max() > 1.0 or predicted_bits.min() < 0.0:
        predicted_bits = (predicted_bits > 0.5).float()

    correct = (predicted_bits == target_bits).float().sum()
    total = target_bits.numel()
    acc = (correct / total * 100).item()
    return acc


def calculate_message_accuracy(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor
) -> float:
    """
    Calculate Message-level Accuracy (exact match of entire message).
    Args:
        predicted_bits: [B, msg_len] predicted bits
        target_bits: [B, msg_len] ground truth bits
    Returns:
        Percentage of messages with zero errors
    """
    if predicted_bits.max() > 1.0 or predicted_bits.min() < 0.0:
        predicted_bits = (predicted_bits > 0.5).float()

    exact_match = (predicted_bits == target_bits).all(dim=1).float().sum()
    total = target_bits.shape[0]
    acc = (exact_match / total * 100).item()
    return acc


def calculate_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    lpips_fn: Optional[callable] = None
) -> float:
    """
    Calculate LPIPS distance.
    Args:
        img1, img2: [B, 3, H, W] in [-1, 1]
        lpips_fn: Pre-initialized LPIPS function
    Returns:
        Mean LPIPS distance
    """
    if lpips_fn is not None:
        with torch.no_grad():
            return lpips_fn(img1, img2).mean().item()
    else:
        # Fallback to MSE
        return F.mse_loss(img1, img2).item()


def evaluate_stego_quality(
    cover: torch.Tensor,
    stego: torch.Tensor,
    message: torch.Tensor,
    decoded_logits: torch.Tensor,
    lpips_fn: Optional[callable] = None
) -> dict:
    """
    Comprehensive evaluation of stego quality and message recovery.
    Returns dictionary with all metrics.
    """
    # Image quality
    psnr = calculate_psnr(cover, stego)
    ssim = calculate_ssim(cover, stego)
    lpips_val = calculate_lpips(cover, stego, lpips_fn)

    # Message recovery
    predicted_bits = (torch.sigmoid(decoded_logits) > 0.5).float()
    ber = calculate_ber(predicted_bits, message)
    bit_acc = calculate_bit_accuracy(predicted_bits, message)
    msg_acc = calculate_message_accuracy(predicted_bits, message)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "ber": ber,
        "bit_accuracy": bit_acc,
        "message_accuracy": msg_acc,
    }


def evaluate_secret_reconstruction(
    secret: torch.Tensor,
    revealed: torch.Tensor,
    lpips_fn: Optional[callable] = None
) -> dict:
    """
    Evaluate secret image reconstruction quality.
    """
    psnr = calculate_psnr(secret, revealed)
    ssim = calculate_ssim(secret, revealed)
    lpips_val = calculate_lpips(secret, revealed, lpips_fn)
    l1 = F.l1_loss(revealed, secret).item()
    mse = F.mse_loss(revealed, secret).item()

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "l1": l1,
        "mse": mse,
    }


class AverageMeter:
    """Track average and current value of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0