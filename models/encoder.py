"""
StegoEncoder - U-Net based encoder that fuses cover image with message
Shared between text and image steganography branches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
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
    """Downsampling block: ResBlock + Strided Conv."""
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
    """Upsampling block: Transposed Conv + ResBlock + Skip Connection."""
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
            # Handle potential size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        if self.use_residual:
            x = self.resblock(x)
        return x


class MessageBroadcaster(nn.Module):
    """
    Broadcast message embedding to spatial dimensions.
    Text: [B, msg_len] -> [B, msg_channels, H, W]
    Image: [B, C, H_s, W_s] -> [B, msg_channels, H, W]
    """
    def __init__(
        self,
        message_length: int,
        embed_dim: int,
        output_channels: int,
        cover_size: int,
        mode: str = "text"  # "text" or "image"
    ):
        super().__init__()
        self.mode = mode
        self.cover_size = cover_size
        self.output_channels = output_channels

        if mode == "text":
            # Text: Embed each bit -> project to spatial map
            self.embedding = nn.Embedding(2, embed_dim)  # 0/1 bits
            self.fc = nn.Linear(message_length * embed_dim, cover_size * cover_size)
            self.conv = nn.Conv2d(1, output_channels, 1)
        else:
            # Image: Encode secret image -> project to message channels
            self.secret_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(inplace=True),  # 64x64
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(inplace=True), # 32x32
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(inplace=True), # 16x16
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, cover_size * cover_size),
            )
            self.conv = nn.Conv2d(1, output_channels, 1)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        B = message.shape[0]
        if self.mode == "text":
            # message: [B, msg_len] with values 0/1
            emb = self.embedding(message.long())  # [B, msg_len, embed_dim]
            emb = emb.view(B, -1)  # [B, msg_len * embed_dim]
            spatial = self.fc(emb)  # [B, H*W]
            spatial = spatial.view(B, 1, self.cover_size, self.cover_size)
        else:
            # message: [B, 3, H_s, W_s] secret image
            spatial = self.secret_encoder(message)  # [B, H*W]
            spatial = spatial.view(B, 1, self.cover_size, self.cover_size)

        out = self.conv(spatial)  # [B, output_channels, H, W]
        return out


class StegoEncoder(nn.Module):
    """
    U-Net Encoder for Steganography.
    Input: Cover image [B, 3, H, W] + Message broadcast [B, msg_ch, H, W]
    Output: Stego image [B, 3, H, W]
    """
    def __init__(
        self,
        cover_size: int = 256,
        message_length: int = 100,
        embed_dim: int = 64,
        base_channels: int = 64,
        depth: int = 5,
        message_channels: int = 32,
        use_residual: bool = True,
        mode: str = "text"
    ):
        super().__init__()
        self.cover_size = cover_size
        self.depth = depth
        self.base_channels = base_channels
        self.mode = mode

        # Message broadcaster
        self.msg_broadcaster = MessageBroadcaster(
            message_length=message_length,
            embed_dim=embed_dim,
            output_channels=message_channels,
            cover_size=cover_size,
            mode=mode
        )

        # Initial convolution: cover(3) + message(message_channels) -> base_channels
        in_ch = 3 + message_channels
        self.init_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        self.init_act = nn.SiLU(inplace=True)

        # Encoder path (downsampling)
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth - 1):
            out_ch = channels * 2
            self.down_blocks.append(DownBlock(channels, out_ch, use_residual))
            channels = out_ch

        # Bottleneck
        self.bottleneck = ResidualBlock(channels) if use_residual else nn.Identity()

        # Decoder path (upsampling)
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            out_ch = channels // 2
            self.up_blocks.append(UpBlock(channels, out_ch, use_residual))
            channels = out_ch

        # Final output: channels -> 3 (stego image)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Store skip connections
        self.skip_connections: List[torch.Tensor] = []

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover: [B, 3, H, W] in range [-1, 1]
            message: Text mode: [B, msg_len] bits (0/1)
                     Image mode: [B, 3, H_s, W_s] in range [-1, 1]
        Returns:
            stego: [B, 3, H, W] in range [-1, 1]
        """
        # Broadcast message to spatial map
        msg_map = self.msg_broadcaster(message)  # [B, msg_ch, H, W]

        # Concatenate cover and message map
        x = torch.cat([cover, msg_map], dim=1)  # [B, 3+msg_ch, H, W]

        # Initial conv
        x = self.init_conv(x)
        x = self.init_act(x)

        # Encoder with skip connections
        self.skip_connections = []
        for down_block in self.down_blocks:
            self.skip_connections.append(x)
            x = down_block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, up_block in enumerate(self.up_blocks):
            skip = self.skip_connections[-(i + 1)]
            x = up_block(x, skip)

        # Final output
        stego = self.final_conv(x)
        return stego


def create_encoder(config: dict, mode: str = "text") -> StegoEncoder:
    """Factory function to create encoder from config."""
    if mode == "text":
        return StegoEncoder(
            cover_size=config["model"]["cover_size"],
            message_length=config["model"]["message_length"],
            embed_dim=config["model"]["embed_dim"],
            base_channels=config["model"]["base_channels"],
            depth=config["model"]["encoder_depth"],
            use_residual=config["model"]["use_residual_blocks"],
            mode="text"
        )
    else:
        return StegoEncoder(
            cover_size=config["model"]["cover_size"],
            message_length=config["model"]["secret_size"] * config["model"]["secret_size"] * 3,
            embed_dim=config["model"]["embed_dim"],
            base_channels=config["model"]["base_channels"],
            depth=config["model"]["encoder_depth"],
            use_residual=config["model"]["use_residual_blocks"],
            mode="image"
        )