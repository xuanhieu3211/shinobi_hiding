"""
PatchGAN Discriminator with Spectral Normalization.
Used for adversarial loss to make stego images indistinguishable from cover.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SpectralConv2d(nn.Module):
    """Conv2d with Spectral Normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field by default).
    Classifies whether each 70x70 patch is real (cover) or fake (stego).
    Uses SpectralNorm for stability on 8GB VRAM.
    """
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        patch_size: int = 70,
        use_sigmoid: bool = False  # False for LSGAN, True for vanilla GAN
    ):
        super().__init__()
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid

        # Layer 1: No norm
        layers = [
            SpectralConv2d(input_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Middle layers: SpectralNorm + LeakyReLU
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                SpectralConv2d(base_channels * nf_mult_prev, base_channels * nf_mult, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Final layer: 1x1 output per patch
        layers += [
            SpectralConv2d(base_channels * nf_mult, 1, 4, stride=1, padding=1)
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] image in range [-1, 1]
        Returns:
            patch_logits: [B, 1, H_out, W_out] - logits for each patch
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (optional upgrade if AUC > 0.7).
    Runs discriminator at multiple scales for better gradient flow.
    """
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        num_scales: int = 3
    ):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(input_channels, base_channels, n_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list:
        """
        Returns:
            List of patch logits at different scales
        """
        results = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            results.append(disc(x))
        return results


def create_discriminator(config: dict) -> PatchGANDiscriminator:
    """Factory function."""
    disc_config = config["discriminator"]
    return PatchGANDiscriminator(
        input_channels=disc_config["input_channels"],
        base_channels=disc_config["base_channels"],
        n_layers=disc_config["n_layers"],
        patch_size=disc_config["patch_size"],
        use_sigmoid=False  # LSGAN
    )