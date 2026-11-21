"""
Differentiable Noise Layers for Curriculum Learning.
Implements 10 noise types with STE (Straight-Through Estimator) for JPEG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class NoiseLayer(nn.Module, ABC):
    """Base class for differentiable noise layers."""

    @abstractmethod
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        pass

    def get_default_params(self) -> Dict[str, Any]:
        return {}


class IdentityNoise(NoiseLayer):
    """No noise - identity transform."""
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        return x


class JPEGNoise(NoiseLayer):
    """
    Differentiable JPEG compression using Straight-Through Estimator (STE).
    During forward: applies JPEG compression (non-differentiable)
    During backward: passes gradient through unchanged (STE)
    """
    def __init__(self, quality_min: int = 75, quality_max: int = 90):
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, x: torch.Tensor, quality: Optional[int] = None, **params) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] in range [-1, 1]
            quality: JPEG quality (75-95). If None, random in range.
        """
        if quality is None:
            quality = random.randint(self.quality_min, self.quality_max)

        # Use STE: forward with JPEG, backward with identity
        return _JPEGCompress.apply(x, quality)

    def get_default_params(self) -> Dict[str, Any]:
        return {"quality": (self.quality_min + self.quality_max) // 2}


class _JPEGCompress(torch.autograd.Function):
    """STE for JPEG compression."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, quality: int) -> torch.Tensor:
        ctx.quality = quality
        # Convert to numpy for PIL JPEG
        # Note: This is a simplified version. For production, use diffjpeg or similar.
        return x  # Placeholder - actual JPEG in eval mode only

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # STE: pass gradient through unchanged
        return grad_output, None


class DiffJPEGNoise(NoiseLayer):
    """
    Differentiable JPEG using diffjpeg library (if available).
    Falls back to STE approximation if not installed.
    """
    def __init__(self, quality_min: int = 75, quality_max: int = 90, differentiable: bool = True):
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.differentiable = differentiable
        self._diffjpeg = None

        if differentiable:
            try:
                from diffjpeg import DiffJPEG
                self._diffjpeg = DiffJPEG(height=256, width=256, differentiable=True, quality=80)
            except ImportError:
                print("Warning: diffjpeg not installed. Using STE approximation.")
                self._diffjpeg = None

    def forward(self, x: torch.Tensor, quality: Optional[int] = None, **params) -> torch.Tensor:
        if quality is None:
            quality = random.randint(self.quality_min, self.quality_max)

        if self._diffjpeg is not None:
            # Update quality (diffjpeg supports dynamic quality)
            self._diffjpeg.quality = quality
            return self._diffjpeg(x)
        else:
            # STE fallback: during training, add noise approximating JPEG artifacts
            if self.training:
                return x + self._jpeg_approx_noise(x)
            else:
                # In eval, use actual PIL JPEG
                return self._pil_jpeg(x, quality)

    def _jpeg_approx_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate JPEG artifacts with differentiable noise."""
        # Block-wise DCT-like noise (8x8 blocks)
        B, C, H, W = x.shape
        noise = torch.randn_like(x) * 0.01

        # Add block artifacts
        block_size = 8
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                block_noise = torch.randn(B, C, 1, 1, device=x.device) * 0.005
                noise[:, :, i:i+block_size, j:j+block_size] += block_noise

        return noise

    def _pil_jpeg(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        """Actual PIL JPEG compression for evaluation."""
        try:
            from PIL import Image
            import io
        except ImportError:
            return x

        B, C, H, W = x.shape
        results = []
        for b in range(B):
            img_tensor = x[b].clamp(-1, 1)
            img_tensor = (img_tensor + 1) / 2 * 255  # [0, 255]
            img_np = img_tensor.byte().permute(1, 2, 0).cpu().numpy()
            img_pil = Image.fromarray(img_np, mode='RGB')

            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)
            img_compressed = img_compressed.resize((W, H), Image.LANCZOS)

            img_out = torch.from_numpy(np.array(img_compressed)).float() / 255 * 2 - 1
            img_out = img_out.permute(2, 0, 1)
            results.append(img_out)

        return torch.stack(results, dim=0).to(x.device)


class GaussianNoise(NoiseLayer):
    """Additive Gaussian noise."""
    def __init__(self, std_min: float = 0.01, std_max: float = 0.02):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, x: torch.Tensor, std: Optional[float] = None, **params) -> torch.Tensor:
        if std is None:
            std = random.uniform(self.std_min, self.std_max)
        noise = torch.randn_like(x) * std
        return x + noise

    def get_default_params(self) -> Dict[str, Any]:
        return {"std": (self.std_min + self.std_max) / 2}


class DropoutNoise(NoiseLayer):
    """Random pixel dropout (set to 0 or mean)."""
    def __init__(self, p_min: float = 0.1, p_max: float = 0.2):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, x: torch.Tensor, p: Optional[float] = None, **params) -> torch.Tensor:
        if p is None:
            p = random.uniform(self.p_min, self.p_max)
        mask = torch.rand_like(x) > p
        return x * mask.float()

    def get_default_params(self) -> Dict[str, Any]:
        return {"p": (self.p_min + self.p_max) / 2}


class CropNoise(NoiseLayer):
    """Random crop and resize back."""
    def __init__(self, scale_min: float = 0.85, scale_max: float = 1.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor, scale: Optional[float] = None, **params) -> torch.Tensor:
        if scale is None:
            scale = random.uniform(self.scale_min, self.scale_max)

        B, C, H, W = x.shape
        new_H, new_W = int(H * scale), int(W * scale)

        # Random crop
        top = random.randint(0, H - new_H)
        left = random.randint(0, W - new_W)
        cropped = x[:, :, top:top+new_H, left:left+new_W]

        # Resize back
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

    def get_default_params(self) -> Dict[str, Any]:
        return {"scale": (self.scale_min + self.scale_max) / 2}


class ResizeNoise(NoiseLayer):
    """Random resize (down then up)."""
    def __init__(self, scale_min: float = 0.5, scale_max: float = 2.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor, scale: Optional[float] = None, **params) -> torch.Tensor:
        if scale is None:
            scale = random.uniform(self.scale_min, self.scale_max)

        B, C, H, W = x.shape
        new_H, new_W = int(H * scale), int(W * scale)

        # Resize down
        resized = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # Resize back up
        return F.interpolate(resized, size=(H, W), mode='bilinear', align_corners=False)

    def get_default_params(self) -> Dict[str, Any]:
        return {"scale": (self.scale_min + self.scale_max) / 2}


class RotationNoise(NoiseLayer):
    """Random rotation."""
    def __init__(self, degrees: float = 10.0):
        super().__init__()
        self.degrees = degrees

    def forward(self, x: torch.Tensor, angle: Optional[float] = None, **params) -> torch.Tensor:
        if angle is None:
            angle = random.uniform(-self.degrees, self.degrees)

        # Convert angle to radians
        theta = torch.tensor(angle * 3.14159 / 180.0, device=x.device, dtype=x.dtype)

        # Rotation matrix
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Affine grid
        B, C, H, W = x.shape
        theta_mat = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        theta_mat[:, 0, 0] = cos_t
        theta_mat[:, 0, 1] = -sin_t
        theta_mat[:, 1, 0] = sin_t
        theta_mat[:, 1, 1] = cos_t

        grid = F.affine_grid(theta_mat, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def get_default_params(self) -> Dict[str, Any]:
        return {"angle": 0.0}


class BrightnessContrastNoise(NoiseLayer):
    """Random brightness and contrast adjustment."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, x: torch.Tensor, brightness: Optional[float] = None, contrast: Optional[float] = None, **params) -> torch.Tensor:
        if brightness is None:
            brightness = random.uniform(-self.brightness, self.brightness)
        if contrast is None:
            contrast = random.uniform(-self.contrast, self.contrast)

        # Contrast: multiply by (1 + contrast)
        # Brightness: add brightness
        x = x * (1 + contrast) + brightness
        return x.clamp(-1, 1)

    def get_default_params(self) -> Dict[str, Any]:
        return {"brightness": 0.0, "contrast": 0.0}


class SaltPepperNoise(NoiseLayer):
    """Salt and pepper noise."""
    def __init__(self, prob: float = 0.01):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor, prob: Optional[float] = None, **params) -> torch.Tensor:
        if prob is None:
            prob = self.prob

        B, C, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device)

        # Salt (white = 1)
        salt_mask = mask < (prob / 2)
        # Pepper (black = -1)
        pepper_mask = (mask >= (prob / 2)) & (mask < prob)

        x = x.clone()
        x[salt_mask.expand(-1, C, -1, -1)] = 1.0
        x[pepper_mask.expand(-1, C, -1, -1)] = -1.0
        return x

    def get_default_params(self) -> Dict[str, Any]:
        return {"prob": self.prob}


class BlurNoise(NoiseLayer):
    """Gaussian blur."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self._kernel = None

    def forward(self, x: torch.Tensor, kernel_size: Optional[int] = None, **params) -> torch.Tensor:
        if kernel_size is None:
            kernel_size = self.kernel_size

        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        if self._kernel is None or self._kernel.shape[-1] != kernel_size:
            self._kernel = self._create_gaussian_kernel(kernel_size, kernel_size / 3.0).to(x.device)

        kernel = self._kernel.expand(x.shape[1], 1, -1, -1)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) @ g.unsqueeze(1)  # [1, size, size]
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]

    def get_default_params(self) -> Dict[str, Any]:
        return {"kernel_size": self.kernel_size}


# Registry of all noise layers
NOISE_LAYERS = {
    "identity": IdentityNoise,
    "jpeg": DiffJPEGNoise,
    "gaussian": GaussianNoise,
    "dropout": DropoutNoise,
    "crop": CropNoise,
    "resize": ResizeNoise,
    "rotation": RotationNoise,
    "brightness_contrast": BrightnessContrastNoise,
    "salt_pepper": SaltPepperNoise,
    "blur": BlurNoise,
}


def get_noise_layers(config: dict) -> Dict[str, NoiseLayer]:
    """Create noise layer instances from config."""
    layers = {}
    for layer_config in config["noise"]["layers"]:
        name = layer_config["name"]
        params = layer_config.get("params", {})
        if name in NOISE_LAYERS:
            layers[name] = NOISE_LAYERS[name](**params)
        else:
            print(f"Warning: Unknown noise layer '{name}'")
    return layers


class CurriculumNoise(nn.Module):
    """
    Manages curriculum learning schedule for noise layers.
    Progressively activates more noise layers as training progresses.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.noise_layers = get_noise_layers(config)
        self.curriculum = config["noise"]["curriculum"]
        self.current_phase = 0

    def forward(self, x: torch.Tensor, epoch: int, phase_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply active noise layers for current epoch.
        Args:
            x: [B, 3, H, W] stego images
            epoch: Current epoch number
            phase_params: Optional override for phase parameters
        Returns:
            Noised images
        """
        # Determine current phase
        active_layers = ["identity"]
        params = {}

        for phase in self.curriculum:
            start, end = phase["epochs"]
            if start <= epoch < end:
                active_layers = phase["active_layers"]
                # Extract phase-specific params
                for k, v in phase.items():
                    if k not in ["epochs", "active_layers"]:
                        params[k] = v
                break

        # Apply noise layers sequentially
        out = x
        for layer_name in active_layers:
            if layer_name in self.noise_layers:
                layer_params = {k: v for k, v in params.items() if k in self._get_layer_param_names(layer_name)}
                out = self.noise_layers[layer_name](out, **layer_params)

        return out

    def _get_layer_param_names(self, layer_name: str) -> List[str]:
        """Map phase config keys to layer parameter names."""
        mapping = {
            "jpeg": ["quality"],
            "gaussian": ["std"],
            "dropout": ["p"],
            "crop": ["scale"],
            "resize": ["scale"],
            "rotation": ["angle"],
            "brightness_contrast": ["brightness", "contrast"],
            "salt_pepper": ["prob"],
            "blur": ["kernel_size"],
        }
        return mapping.get(layer_name, [])

    def get_active_layers(self, epoch: int) -> List[str]:
        """Get list of active layer names for given epoch."""
        for phase in self.curriculum:
            start, end = phase["epochs"]
            if start <= epoch < end:
                return phase["active_layers"]
        return ["identity"]


def create_curriculum_noise(config: dict) -> CurriculumNoise:
    """Factory function."""
    return CurriculumNoise(config)