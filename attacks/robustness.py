"""
Robustness Testing for Steganography.
Tests message recovery under various image transformations.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import calculate_ber, calculate_bit_accuracy, calculate_message_accuracy


class NoiseTransform:
    """Base class for noise transforms."""
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class JPEGCompression(NoiseTransform):
    """JPEG compression transform."""
    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [3, H, W] in [-1, 1]
        img_pil = self._tensor_to_pil(img)

        from io import BytesIO
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)

        return self._pil_to_tensor(img_compressed)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu()
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
        tensor = (tensor * 255).byte()
        return TF.to_pil_image(tensor)

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(pil_img)  # [0, 1]
        return tensor * 2 - 1  # [-1, 1]


class GaussianNoise(NoiseTransform):
    """Add Gaussian noise."""
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(-1, 1)


class DropoutNoise(NoiseTransform):
    """Random pixel dropout."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(img) > self.p
        return img * mask.float()


class CropAndResize(NoiseTransform):
    """Random crop and resize back."""
    def __init__(self, scale: float = 0.9):
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        new_H, new_W = int(H * self.scale), int(W * self.scale)

        # Random crop
        top = random.randint(0, H - new_H)
        left = random.randint(0, W - new_W)
        cropped = img[:, top:top+new_H, left:left+new_W]

        # Resize back
        return F.interpolate(cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)


class ResizeTransform(NoiseTransform):
    """Resize down and up."""
    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        new_H, new_W = int(H * self.scale), int(W * self.scale)

        resized = F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False)
        return F.interpolate(resized, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)


class RotationTransform(NoiseTransform):
    """Random rotation."""
    def __init__(self, degrees: float = 10.0):
        self.degrees = degrees

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)


class BrightnessContrastTransform(NoiseTransform):
    """Brightness and contrast adjustment."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        b = random.uniform(-self.brightness, self.brightness)
        c = random.uniform(-self.contrast, self.contrast)
        img = img * (1 + c) + b
        return img.clamp(-1, 1)


class SaltPepperNoise(NoiseTransform):
    """Salt and pepper noise."""
    def __init__(self, prob: float = 0.01):
        self.prob = prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(img[:, 0:1, :, :])  # [1, H, W]
        salt = mask < (self.prob / 2)
        pepper = (mask >= (self.prob / 2)) & (mask < self.prob)

        img = img.clone()
        img[:, salt.expand(3, -1, -1)] = 1.0
        img[:, pepper.expand(3, -1, -1)] = -1.0
        return img


class BlurTransform(NoiseTransform):
    """Gaussian blur."""
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel = self._create_gaussian_kernel(kernel_size)

    def _create_gaussian_kernel(self, size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * (size / 3.0) ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) @ g.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(img.device).expand(3, 1, -1, -1)
        padding = self.kernel_size // 2
        return F.conv2d(img.unsqueeze(0), kernel, padding=padding, groups=3).squeeze(0)


# Registry of all transforms
ROBUSTNESS_TRANSFORMS = {
    "jpeg": JPEGCompression,
    "gaussian": GaussianNoise,
    "dropout": DropoutNoise,
    "crop": CropAndResize,
    "resize": ResizeTransform,
    "rotation": RotationTransform,
    "brightness_contrast": BrightnessContrastTransform,
    "salt_pepper": SaltPepperNoise,
    "blur": BlurTransform,
}


class RobustnessTester:
    """
    Tests steganography model robustness against various image transformations.
    """
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def test_single(
        self,
        stego: torch.Tensor,
        message: torch.Tensor,
        transform: NoiseTransform,
        decoder_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Test robustness against a single transform.
        Args:
            stego: [B, 3, H, W] stego images
            message: [B, msg_len] ground truth messages
            transform: NoiseTransform to apply
            decoder_fn: Optional custom decoder function
        Returns:
            Dict with BER, bit accuracy, message accuracy
        """
        if decoder_fn is None:
            decoder_fn = lambda x: self.model.decoder(x) if hasattr(self.model, 'decoder') else self.model(x)

        # Apply transform to each image in batch
        B = stego.shape[0]
        noised = []
        for i in range(B):
            noised.append(transform(stego[i]))
        noised = torch.stack(noised, dim=0).to(self.device)

        # Decode
        with torch.no_grad():
            logits = decoder_fn(noised)

        # Calculate metrics
        ber = calculate_ber(logits, message)
        bit_acc = calculate_bit_accuracy(logits, message)
        msg_acc = calculate_message_accuracy(logits, message)

        return {
            "ber": ber,
            "bit_accuracy": bit_acc,
            "message_accuracy": msg_acc,
        }

    def test_all(
        self,
        stego: torch.Tensor,
        message: torch.Tensor,
        transforms_config: Dict[str, Dict],
        decoder_fn: Optional[Callable] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Test robustness against all configured transforms.
        Args:
            stego: [B, 3, H, W] stego images
            message: [B, msg_len] ground truth messages
            transforms_config: Dict of transform_name -> params
            decoder_fn: Optional custom decoder function
        Returns:
            Dict of transform_name -> metrics
        """
        results = {}
        for name, params in transforms_config.items():
            if name in ROBUSTNESS_TRANSFORMS:
                transform = ROBUSTNESS_TRANSFORMS[name](**params)
                metrics = self.test_single(stego, message, transform, decoder_fn)
                results[name] = metrics
                print(f"  {name}: BER={metrics['ber']:.2f}%, BitAcc={metrics['bit_accuracy']:.2f}%, MsgAcc={metrics['message_accuracy']:.2f}%")
            else:
                print(f"  Warning: Unknown transform '{name}'")
        return results


def test_robustness(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    transforms_config: Dict[str, Dict],
    device: Optional[torch.device] = None,
    max_batches: int = 10,
    decoder_fn: Optional[Callable] = None
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive robustness test on test dataset.
    """
    tester = RobustnessTester(model, device)
    all_results = {name: {"ber": [], "bit_accuracy": [], "message_accuracy": []}
                   for name in transforms_config.keys()}

    model.eval()
    with torch.no_grad():
        for batch_idx, (cover, message) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            cover = cover.to(device)
            message = message.to(device)

            # Generate stego
            if hasattr(model, 'encoder'):
                stego = model.encoder(cover, message)
            else:
                stego = model(cover, message)

            # Test each transform
            for name, params in transforms_config.items():
                if name in ROBUSTNESS_TRANSFORMS:
                    transform = ROBUSTNESS_TRANSFORMS[name](**params)

                    # Apply to batch
                    B = stego.shape[0]
                    noised = []
                    for i in range(B):
                        noised.append(transform(stego[i]))
                    noised = torch.stack(noised, dim=0).to(device)

                    # Decode
                    if decoder_fn:
                        logits = decoder_fn(noised)
                    elif hasattr(model, 'decoder'):
                        logits = model.decoder(noised)
                    else:
                        _, logits = model(noised, message)  # Assuming forward returns (stego, logits)

                    # Metrics
                    ber = calculate_ber(logits, message)
                    bit_acc = calculate_bit_accuracy(logits, message)
                    msg_acc = calculate_message_accuracy(logits, message)

                    all_results[name]["ber"].append(ber)
                    all_results[name]["bit_accuracy"].append(bit_acc)
                    all_results[name]["message_accuracy"].append(msg_acc)

    # Average results
    final_results = {}
    for name, metrics in all_results.items():
        final_results[name] = {
            "ber": np.mean(metrics["ber"]) if metrics["ber"] else 0,
            "bit_accuracy": np.mean(metrics["bit_accuracy"]) if metrics["bit_accuracy"] else 0,
            "message_accuracy": np.mean(metrics["message_accuracy"]) if metrics["message_accuracy"] else 0,
            "std_ber": np.std(metrics["ber"]) if metrics["ber"] else 0,
        }

    return final_results


def create_default_transforms_config() -> Dict[str, Dict]:
    """Create default robustness test configuration."""
    return {
        "jpeg_90": {"quality": 90},
        "jpeg_75": {"quality": 75},
        "jpeg_50": {"quality": 50},
        "gaussian_001": {"std": 0.01},
        "gaussian_002": {"std": 0.02},
        "dropout_01": {"p": 0.1},
        "dropout_02": {"p": 0.2},
        "crop_09": {"scale": 0.9},
        "crop_085": {"scale": 0.85},
        "resize_05": {"scale": 0.5},
        "resize_2": {"scale": 2.0},
        "rotation_5": {"degrees": 5},
        "rotation_10": {"degrees": 10},
        "brightness_contrast": {"brightness": 0.2, "contrast": 0.2},
        "salt_pepper": {"prob": 0.01},
        "blur": {"kernel_size": 3},
    }