"""
TextDecoder - Decodes hidden message bits from stego image.
Uses 1x1 Conv + Flatten + FC (or Transformer) to preserve spatial information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TextDecoderConvFC(nn.Module):
    """
    Convolutional + FC decoder for text steganography.
    Preserves spatial correlation between bits via 1x1 conv before flattening.
    """
    def __init__(
        self,
        input_channels: int = 3,
        message_length: int = 100,
        base_channels: int = 64,
        depth: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.message_length = message_length

        # Feature extraction: progressively reduce spatial, increase channels
        layers = []
        in_ch = input_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(8, out_ch))
            layers.append(nn.SiLU(inplace=True))
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_channels = in_ch

        # 1x1 conv to mix spatial information before pooling
        self.spatial_mixer = nn.Conv2d(self.feature_channels, self.feature_channels, 1)

        # Global pooling + FC head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, message_length)
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego: [B, 3, H, W] in range [-1, 1]
        Returns:
            logits: [B, message_length] - raw logits for each bit (BCEWithLogitsLoss)
        """
        B = stego.shape[0]
        x = self.feature_extractor(stego)  # [B, C, H', W']
        x = self.spatial_mixer(x)          # [B, C, H', W'] - mix spatial info
        x = self.pool(x).view(B, -1)       # [B, C]
        logits = self.fc(x)                # [B, message_length]
        return logits


class TextDecoderTransformer(nn.Module):
    """
    Transformer-based decoder for text steganography.
    Treats spatial patches as sequence tokens for better bit correlation modeling.
    """
    def __init__(
        self,
        input_channels: int = 3,
        message_length: int = 100,
        base_channels: int = 64,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.message_length = message_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Initial conv to reduce channels and extract patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, embed_dim, patch_size, stride=patch_size),
        )

        # Positional embedding (learned)
        num_patches = (256 // patch_size) ** 2  # 16x16 = 256 patches for 256x256
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: pool over patches -> message bits
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, message_length)
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego: [B, 3, H, W] in range [-1, 1]
        Returns:
            logits: [B, message_length]
        """
        B = stego.shape[0]

        # Patch embedding: [B, embed_dim, H/p, W/p] -> [B, num_patches, embed_dim]
        x = self.patch_embed(stego)  # [B, embed_dim, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)  # [B, 256, embed_dim]

        # Global average pooling over patches
        x = x.mean(dim=1)  # [B, embed_dim]

        # Output logits
        logits = self.output_head(x)  # [B, message_length]
        return logits


class TextDecoder(nn.Module):
    """
    Unified Text Decoder supporting both ConvFC and Transformer backends.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.message_length = config["model"]["message_length"]
        self.decoder_type = config["model"].get("decoder_type", "conv_fc")

        if self.decoder_type == "transformer":
            self.decoder = TextDecoderTransformer(
                input_channels=3,
                message_length=self.message_length,
                base_channels=config["model"]["base_channels"],
                num_layers=config["model"].get("transformer_layers", 2),
                num_heads=config["model"].get("transformer_heads", 4),
            )
        else:
            self.decoder = TextDecoderConvFC(
                input_channels=3,
                message_length=self.message_length,
                base_channels=config["model"]["base_channels"],
                depth=4,
                hidden_dim=512
            )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego: [B, 3, H, W] in range [-1, 1]
        Returns:
            logits: [B, message_length] - raw logits for BCEWithLogitsLoss
        """
        return self.decoder(stego)

    def decode_bits(self, stego: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Decode bits from stego image.
        Args:
            stego: [B, 3, H, W]
            threshold: logit threshold (0.0 = sigmoid(0) = 0.5)
        Returns:
            bits: [B, message_length] with values 0/1
        """
        logits = self.forward(stego)
        probs = torch.sigmoid(logits)
        bits = (probs > 0.5).float()
        return bits

    def bits_to_text(self, bits: torch.Tensor, encoding: str = "ascii") -> list:
        """
        Convert bit tensor to text strings.
        Args:
            bits: [B, message_length] with values 0/1
        Returns:
            List of decoded strings
        """
        results = []
        for b in bits:
            # Convert bits to bytes
            bit_str = ''.join(str(int(x.item())) for x in b)
            # Pad to multiple of 8
            padding = (8 - len(bit_str) % 8) % 8
            bit_str = bit_str + '0' * padding
            # Convert to bytes
            byte_array = bytearray()
            for i in range(0, len(bit_str), 8):
                byte_val = int(bit_str[i:i+8], 2)
                if byte_val == 0:
                    break
                byte_array.append(byte_val)
            try:
                text = byte_array.decode(encoding, errors='ignore')
            except:
                text = ""
            results.append(text)
        return results


def create_text_decoder(config: dict) -> TextDecoder:
    """Factory function."""
    return TextDecoder(config)