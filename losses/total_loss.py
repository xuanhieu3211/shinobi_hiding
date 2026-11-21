"""
Total Loss Functions combining all components for Text and Image Steganography.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .lpips_loss import LPIPSLoss
from .adversarial_loss import AdversarialLoss
from .reconstruction_loss import ReconstructionLoss, ImageReconstructionLoss, CoverReconstructionLoss


class TextStegoLoss(nn.Module):
    """
    Total Loss for Text Steganography.

    L_total = λ_pixel * L_pixel + λ_lpips * L_lpips + λ_adv * L_adv + λ_msg * L_msg

    Where:
    - L_pixel: MSE between cover and stego
    - L_lpips: Perceptual distance (LPIPS-AlexNet)
    - L_adv: Adversarial loss (LSGAN) from discriminator
    - L_msg: Message reconstruction loss (BCE per bit)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.lambda_pixel = config["loss"]["lambda_pixel"]
        self.lambda_lpips = config["loss"]["lambda_lpips"]
        self.lambda_adv = config["loss"]["lambda_adv"]
        self.lambda_msg = config["loss"]["lambda_msg"]

        # Loss components
        self.pixel_loss = CoverReconstructionLoss(loss_type="mse")
        self.lpips_loss = LPIPSLoss(
            pretrained_path=config["paths"].get("pretrained_lpips", None)
        )
        self.adv_loss = AdversarialLoss(gan_type="lsgan")
        self.msg_loss = ReconstructionLoss(loss_type="bce")

    def forward(
        self,
        cover: torch.Tensor,
        stego: torch.Tensor,
        noised_stego: torch.Tensor,
        decoded_logits: torch.Tensor,
        message: torch.Tensor,
        disc_real: Optional[torch.Tensor] = None,
        disc_fake: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        Args:
            cover: [B, 3, H, W] original cover images
            stego: [B, 3, H, W] stego images (before noise)
            noised_stego: [B, 3, H, W] stego after noise layers
            decoded_logits: [B, msg_len] decoder output logits
            message: [B, msg_len] ground truth message bits
            disc_real: Discriminator output for cover images (for disc loss)
            disc_fake: Discriminator output for stego images (for gen loss)
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}

        # Pixel loss (cover vs stego)
        losses["pixel"] = self.pixel_loss(stego, cover)

        # Perceptual loss (LPIPS)
        losses["lpips"] = self.lpips_loss(stego, cover)

        # Message reconstruction loss (on noised stego)
        losses["message"] = self.msg_loss(decoded_logits, message)

        # Adversarial loss (generator)
        if disc_fake is not None:
            losses["adv_gen"] = self.adv_loss.generator_loss(disc_fake)
        else:
            losses["adv_gen"] = torch.tensor(0.0, device=cover.device)

        # Total generator loss
        losses["total_gen"] = (
            self.lambda_pixel * losses["pixel"] +
            self.lambda_lpips * losses["lpips"] +
            self.lambda_msg * losses["message"] +
            self.lambda_adv * losses["adv_gen"]
        )

        # Discriminator loss (if provided)
        if disc_real is not None and disc_fake is not None:
            disc_total, disc_real_loss, disc_fake_loss = self.adv_loss.discriminator_loss(
                disc_real, disc_fake.detach()  # Detach to not backprop through generator
            )
            losses["disc_total"] = disc_total
            losses["disc_real"] = disc_real_loss
            losses["disc_fake"] = disc_fake_loss
        else:
            losses["disc_total"] = torch.tensor(0.0, device=cover.device)

        return losses

    def generator_loss(self, *args, **kwargs) -> torch.Tensor:
        """Convenience method returning only total generator loss."""
        return self.forward(*args, **kwargs)["total_gen"]


class ImageStegoLoss(nn.Module):
    """
    Total Loss for Image Steganography (Supplementary Branch).

    L_total = λ_pixel * L_pixel + λ_secret * L_secret +
              λ_lpips_cover * L_lpips_cover + λ_lpips_secret * L_lpips_secret +
              λ_adv * L_adv

    Where:
    - L_pixel: L1 between cover and stego
    - L_secret: L1 + LPIPS between secret and revealed_secret
    - L_lpips_cover: LPIPS between cover and stego
    - L_lpips_secret: LPIPS between secret and revealed_secret
    - L_adv: Adversarial loss
    """
    def __init__(self, config: dict):
        super().__init__()
        self.lambda_pixel = config["loss"]["lambda_pixel"]
        self.lambda_secret = config["loss"]["lambda_secret"]
        self.lambda_lpips_cover = config["loss"]["lambda_lpips_cover"]
        self.lambda_lpips_secret = config["loss"]["lambda_lpips_secret"]
        self.lambda_adv = config["loss"]["lambda_adv"]

        # Loss components
        self.pixel_loss = CoverReconstructionLoss(loss_type="l1")
        self.lpips_loss = LPIPSLoss(
            pretrained_path=config["paths"].get("pretrained_lpips", None)
        )
        self.secret_loss = ImageReconstructionLoss(use_lpips=True, lpips_weight=1.0)
        self.adv_loss = AdversarialLoss(gan_type="lsgan")

    def forward(
        self,
        cover: torch.Tensor,
        stego: torch.Tensor,
        noised_stego: torch.Tensor,
        secret: torch.Tensor,
        revealed_secret: torch.Tensor,
        disc_real: Optional[torch.Tensor] = None,
        disc_fake: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        Args:
            cover: [B, 3, 256, 256] original cover
            stego: [B, 3, 256, 256] stego (before noise)
            noised_stego: [B, 3, 256, 256] stego after noise
            secret: [B, 3, 128, 128] original secret
            revealed_secret: [B, 3, 128, 128] decoded secret
            disc_real: Discriminator output for cover
            disc_fake: Discriminator output for stego
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}

        # Cover reconstruction (stego vs cover)
        losses["pixel"] = self.pixel_loss(stego, cover)
        losses["lpips_cover"] = self.lpips_loss(stego, cover)

        # Secret reconstruction (revealed vs secret)
        secret_total, secret_l1, secret_lpips = self.secret_loss(revealed_secret, secret)
        losses["secret_l1"] = secret_l1
        losses["secret_lpips"] = secret_lpips
        losses["secret"] = secret_total

        # Adversarial loss (generator)
        if disc_fake is not None:
            losses["adv_gen"] = self.adv_loss.generator_loss(disc_fake)
        else:
            losses["adv_gen"] = torch.tensor(0.0, device=cover.device)

        # Total generator loss
        losses["total_gen"] = (
            self.lambda_pixel * losses["pixel"] +
            self.lambda_lpips_cover * losses["lpips_cover"] +
            self.lambda_secret * losses["secret"] +
            self.lambda_adv * losses["adv_gen"]
        )

        # Discriminator loss
        if disc_real is not None and disc_fake is not None:
            disc_total, disc_real_loss, disc_fake_loss = self.adv_loss.discriminator_loss(
                disc_real, disc_fake.detach()
            )
            losses["disc_total"] = disc_total
            losses["disc_real"] = disc_real_loss
            losses["disc_fake"] = disc_fake_loss
        else:
            losses["disc_total"] = torch.tensor(0.0, device=cover.device)

        return losses


def create_loss(config: dict, mode: str = "text") -> nn.Module:
    """Factory function."""
    if mode == "text":
        return TextStegoLoss(config)
    else:
        return ImageStegoLoss(config)