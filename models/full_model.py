"""
Full Model Wrappers for Text and Image Steganography.
Combines Encoder, Decoder, Discriminator, and Noise Layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from .encoder import StegoEncoder, create_encoder
from .decoder_text import TextDecoder, create_text_decoder
from .decoder_image import ImageDecoder, create_image_decoder
from .discriminator import PatchGANDiscriminator, create_discriminator
from .noise_layers import CurriculumNoise, create_curriculum_noise


class TextStegoModel(nn.Module):
    """
    Complete Text Steganography Model.
    Includes: Encoder, TextDecoder, Discriminator, Noise Layers.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Core components
        self.encoder = create_encoder(config, mode="text")
        self.decoder = create_text_decoder(config)
        self.discriminator = create_discriminator(config)
        self.noise = create_curriculum_noise(config)

        # Loss weights
        self.lambda_pixel = config["loss"]["lambda_pixel"]
        self.lambda_lpips = config["loss"]["lambda_lpips"]
        self.lambda_adv = config["loss"]["lambda_adv"]
        self.lambda_msg = config["loss"]["lambda_msg"]

    def forward(
        self,
        cover: torch.Tensor,
        message: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        Args:
            cover: [B, 3, H, W] cover images in [-1, 1]
            message: [B, msg_len] binary messages (0/1)
            epoch: Current epoch for curriculum noise
        Returns:
            Dict with stego, decoded_logits, noised_stego
        """
        # Encode
        stego = self.encoder(cover, message)

        # Apply curriculum noise
        noised_stego = self.noise(stego, epoch)

        # Decode from noised stego
        decoded_logits = self.decoder(noised_stego)

        return {
            "stego": stego,
            "noised_stego": noised_stego,
            "decoded_logits": decoded_logits
        }

    def encode(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Encode message into cover (inference)."""
        self.eval()
        with torch.no_grad():
            stego = self.encoder(cover, message)
        return stego

    def decode(self, stego: torch.Tensor) -> torch.Tensor:
        """Decode message from stego (inference)."""
        self.eval()
        with torch.no_grad():
            logits = self.decoder(stego)
            bits = (torch.sigmoid(logits) > 0.5).float()
        return bits

    def decode_logits(self, stego: torch.Tensor) -> torch.Tensor:
        """Get raw logits from stego."""
        self.eval()
        with torch.no_grad():
            return self.decoder(stego)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator forward."""
        return self.discriminator(x)


class ImageStegoModel(nn.Module):
    """
    Complete Image Steganography Model (Supplementary Branch).
    Includes: Encoder, ImageDecoder, Discriminator, Noise Layers.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Core components
        self.encoder = create_encoder(config, mode="image")
        self.decoder = create_image_decoder(config)
        self.discriminator = create_discriminator(config)
        self.noise = create_curriculum_noise(config)

        # Loss weights
        self.lambda_pixel = config["loss"]["lambda_pixel"]
        self.lambda_secret = config["loss"]["lambda_secret"]
        self.lambda_lpips_cover = config["loss"]["lambda_lpips_cover"]
        self.lambda_lpips_secret = config["loss"]["lambda_lpips_secret"]
        self.lambda_adv = config["loss"]["lambda_adv"]

    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        Args:
            cover: [B, 3, 256, 256] cover images in [-1, 1]
            secret: [B, 3, 128, 128] secret images in [-1, 1]
            epoch: Current epoch for curriculum noise
        Returns:
            Dict with stego, revealed_secret, noised_stego
        """
        # Encode
        stego = self.encoder(cover, secret)

        # Apply curriculum noise
        noised_stego = self.noise(stego, epoch)

        # Decode secret from noised stego
        revealed_secret = self.decoder(noised_stego)

        return {
            "stego": stego,
            "noised_stego": noised_stego,
            "revealed_secret": revealed_secret
        }

    def encode(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """Encode secret into cover (inference)."""
        self.eval()
        with torch.no_grad():
            stego = self.encoder(cover, secret)
        return stego

    def decode(self, stego: torch.Tensor) -> torch.Tensor:
        """Reveal secret from stego (inference)."""
        self.eval()
        with torch.no_grad():
            revealed = self.decoder(stego)
        return revealed

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator forward."""
        return self.discriminator(x)


def create_model(config: dict, mode: str = "text") -> nn.Module:
    """Factory function to create model."""
    if mode == "text":
        return TextStegoModel(config)
    else:
        return ImageStegoModel(config)