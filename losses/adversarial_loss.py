"""
Adversarial Loss for GAN training.
Supports LSGAN (Least Squares GAN) and vanilla GAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for generator and discriminator.
    LSGAN by default (more stable).
    """
    def __init__(self, gan_type: str = "lsgan", real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.gan_type = gan_type.lower()
        self.real_label = real_label
        self.fake_label = fake_label

        if self.gan_type == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif self.gan_type == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.gan_type == "hinge":
            self.loss_fn = None  # Custom implementation
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def generator_loss(self, disc_fake: torch.Tensor) -> torch.Tensor:
        """
        Generator loss: fool discriminator into thinking fake is real.
        Args:
            disc_fake: Discriminator output for generated (stego) images
        Returns:
            Scalar loss
        """
        if self.gan_type == "lsgan":
            target = torch.full_like(disc_fake, self.real_label)
            return self.loss_fn(disc_fake, target)
        elif self.gan_type == "vanilla":
            target = torch.full_like(disc_fake, self.real_label)
            return self.loss_fn(disc_fake, target)
        elif self.gan_type == "hinge":
            # Hinge loss: -E[D(G(z))]
            return -disc_fake.mean()
        else:
            raise ValueError(f"Unknown GAN type: {self.gan_type}")

    def discriminator_loss(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminator loss: classify real as real, fake as fake.
        Args:
            disc_real: Discriminator output for real (cover) images
            disc_fake: Discriminator output for fake (stego) images
        Returns:
            (total_loss, real_loss, fake_loss)
        """
        if self.gan_type == "lsgan":
            real_target = torch.full_like(disc_real, self.real_label)
            fake_target = torch.full_like(disc_fake, self.fake_label)
            loss_real = self.loss_fn(disc_real, real_target)
            loss_fake = self.loss_fn(disc_fake, fake_target)
            return (loss_real + loss_fake) * 0.5, loss_real, loss_fake

        elif self.gan_type == "vanilla":
            real_target = torch.full_like(disc_real, self.real_label)
            fake_target = torch.full_like(disc_fake, self.fake_label)
            loss_real = self.loss_fn(disc_real, real_target)
            loss_fake = self.loss_fn(disc_fake, fake_target)
            return (loss_real + loss_fake) * 0.5, loss_real, loss_fake

        elif self.gan_type == "hinge":
            loss_real = F.relu(1.0 - disc_real).mean()
            loss_fake = F.relu(1.0 + disc_fake).mean()
            return (loss_real + loss_fake) * 0.5, loss_real, loss_fake
        else:
            raise ValueError(f"Unknown GAN type: {self.gan_type}")


class MultiScaleAdversarialLoss(nn.Module):
    """
    Adversarial loss for multi-scale discriminator.
    Computes loss at each scale and averages.
    """
    def __init__(self, gan_type: str = "lsgan", num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.losses = nn.ModuleList([
            AdversarialLoss(gan_type) for _ in range(num_scales)
        ])

    def generator_loss(self, disc_fake_list: list) -> torch.Tensor:
        total_loss = 0
        for i, disc_fake in enumerate(disc_fake_list):
            total_loss += self.losses[i].generator_loss(disc_fake)
        return total_loss / self.num_scales

    def discriminator_loss(self, disc_real_list: list, disc_fake_list: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_loss = 0
        total_real = 0
        total_fake = 0
        for i, (disc_real, disc_fake) in enumerate(zip(disc_real_list, disc_fake_list)):
            loss, real_l, fake_l = self.losses[i].discriminator_loss(disc_real, disc_fake)
            total_loss += loss
            total_real += real_l
            total_fake += fake_l
        return total_loss / self.num_scales, total_real / self.num_scales, total_fake / self.num_scales


def create_adversarial_loss(config: dict) -> AdversarialLoss:
    """Factory function."""
    # Default to LSGAN for stability
    return AdversarialLoss(gan_type="lsgan")