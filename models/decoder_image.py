"""
ImageDecoder - Symmetric U-Net decoder for image-in-image steganography.
Reconstructs secret image (128x128) from stego image (256x256).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block."""
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        if use_residual:
            self.resblock = ResidualBlock(in_channels)
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            x = self.resblock(x)
        x = self.down(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.act = nn.SiLU(inplace=True)
        self.use_residual = use_residual
        if use_residual:
            self.resblock = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        x = self.act(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        if self.use_residual:
            x = self.resblock(x)
        return x


class ImageDecoder(nn.Module):
    """
    U-Net Decoder for Image Steganography.
    Input: Stego image [B, 3, 256, 256]
    Output: Reconstructed secret [B, 3, 128, 128]
    """
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        output_size: int = 128,
        base_channels: int = 64,
        depth: int = 5,
        use_residual: bool = True
    ):
        super().__init__()
        self.output_size = output_size
        self.depth = depth

        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.init_act = nn.SiLU(inplace=True)

        # Encoder path (downsampling from 256 to bottleneck)
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth - 1):
            out_ch = channels * 2
            self.down_blocks.append(DownBlock(channels, out_ch, use_residual))
            channels = out_ch

        # Bottleneck
        self.bottleneck = ResidualBlock(channels) if use_residual else nn.Identity()

        # Decoder path (upsampling to output_size)
        # We need to go from 256 -> 128 -> 64 -> 32 -> 16 -> 8 (bottleneck)
        # Then 8 -> 16 -> 32 -> 64 -> 128 (output)
        # But our depth=5 gives us: 256 -> 128 -> 64 -> 32 -> 16 (4 down)
        # Then 16 -> 32 -> 64 -> 128 -> 256 (4 up) = output 256
        # We want output 128, so we stop one upsample early

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            out_ch = channels // 2
            self.up_blocks.append(UpBlock(channels, out_ch, use_residual))
            channels = out_ch

        # Final output at 128x128 (after 3 upsamples from 16x16 bottleneck)
        # If depth=5: 256->128->64->32->16 (bottleneck) -> 32->64->128 (3 up)
        # We need exactly 3 up blocks to reach 128 from 16
        # So we'll use only the first 3 up blocks for 128 output

        # Actually let's make it dynamic based on output_size
        self.num_upsamples_to_output = 3  # 16 -> 32 -> 64 -> 128

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, output_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

        self.skip_connections: List[torch.Tensor] = []

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego: [B, 3, 256, 256] in range [-1, 1]
        Returns:
            secret_reconstructed: [B, 3, 128, 128] in range [-1, 1]
        """
        # Initial conv
        x = self.init_conv(stego)
        x = self.init_act(x)

        # Encoder with skip connections
        self.skip_connections = []
        for down_block in self.down_blocks:
            self.skip_connections.append(x)
            x = down_block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder - only use first num_upsamples_to_output up blocks
        for i in range(self.num_upsamples_to_output):
            skip = self.skip_connections[-(i + 1)]
            x = self.up_blocks[i](x, skip)

        # Final output
        secret = self.final_conv(x)  # [B, 3, 128, 128]
        return secret


def create_image_decoder(config: dict) -> ImageDecoder:
    """Factory function."""
    return ImageDecoder(
        input_channels=3,
        output_channels=config["model"]["secret_channels"],
        output_size=config["model"]["secret_size"],
        base_channels=config["model"]["base_channels"],
        depth=config["model"]["encoder_depth"],
        use_residual=config["model"]["use_residual_blocks"]
    )