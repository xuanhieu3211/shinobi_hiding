"""
LPIPS (Learned Perceptual Image Patch Similarity) Loss.
Uses AlexNet backbone with frozen weights for perceptual quality measurement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os


class LPIPSLoss(nn.Module):
    """
    LPIPS Loss using AlexNet backbone.
    Measures perceptual similarity between two images.
    """
    def __init__(
        self,
        net: str = "alex",
        version: str = "0.1",
        pretrained_path: Optional[str] = None,
        normalize: bool = True
    ):
        super().__init__()
        self.normalize = normalize

        # Load LPIPS network
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net, version=version, verbose=False)

            # Load custom weights if provided
            if pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.lpips_fn.load_state_dict(state_dict)
                print(f"Loaded LPIPS weights from {pretrained_path}")

            # Freeze weights
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
            self.lpips_fn.eval()

        except ImportError:
            print("Warning: lpips package not installed. Using MSE fallback.")
            self.lpips_fn = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance between x and y.
        Args:
            x: [B, 3, H, W] in range [-1, 1]
            y: [B, 3, H, W] in range [-1, 1]
        Returns:
            Scalar tensor (mean over batch)
        """
        if self.lpips_fn is not None:
            # LPIPS expects inputs in [-1, 1] range
            if self.normalize:
                # Already in [-1, 1], just ensure
                x = x.clamp(-1, 1)
                y = y.clamp(-1, 1)
            return self.lpips_fn(x, y).mean()
        else:
            # Fallback to MSE
            return F.mse_loss(x, y)

    def to(self, device):
        if self.lpips_fn is not None:
            self.lpips_fn = self.lpips_fn.to(device)
        return super().to(device)


class LPIPSAlexNet(nn.Module):
    """
    Standalone AlexNet-based LPIPS implementation (no external dependency).
    Uses pretrained weights from official LPIPS repository.
    """
    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()
        # AlexNet feature extractor (first 5 layers)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Linear layers for LPIPS (1x1 conv on each feature map)
        self.lin_layers = nn.ModuleList([
            nn.Conv2d(64, 1, 1),
            nn.Conv2d(192, 1, 1),
            nn.Conv2d(384, 1, 1),
            nn.Conv2d(256, 1, 1),
            nn.Conv2d(256, 1, 1),
        ])

        # Scaling factors for each layer (from official LPIPS)
        self.scaling_layers = nn.ModuleList([
            nn.Conv2d(1, 1, 1, bias=False) for _ in range(5)
        ])

        # Initialize scaling layers
        for layer in self.scaling_layers:
            nn.init.constant_(layer.weight, 1.0)

        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
        else:
            # Freeze feature extractor
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()

        # Freeze linear layers (they are learned in original LPIPS but we use pretrained)
        for param in self.lin_layers.parameters():
            param.requires_grad = False
        for param in self.scaling_layers.parameters():
            param.requires_grad = False

    def load_pretrained(self, path: str):
        """Load pretrained LPIPS weights."""
        state_dict = torch.load(path, map_location='cpu')
        # Handle different weight formats
        if 'lin0.weight' in state_dict:
            # Official LPIPS format
            for i in range(5):
                if f'lin{i}.weight' in state_dict:
                    self.lin_layers[i].weight.data = state_dict[f'lin{i}.weight']
                    self.lin_layers[i].bias.data = state_dict.get(f'lin{i}.bias', torch.zeros(1))
                if f'scale{i}.weight' in state_dict:
                    self.scaling_layers[i].weight.data = state_dict[f'scale{i}.weight']
        print(f"Loaded LPIPS weights from {path}")

    def forward_once(self, x: torch.Tensor) -> list:
        """Extract features from all relevant layers."""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [1, 4, 7, 9, 11]:  # After each ReLU/MaxPool block
                features.append(x)
        return features

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance.
        Args:
            x, y: [B, 3, H, W] in range [-1, 1]
        Returns:
            Scalar distance per image (mean over batch)
        """
        B = x.shape[0]

        # Extract features
        feat_x = self.forward_once(x)
        feat_y = self.forward_once(y)

        # Compute weighted L2 distance at each layer
        distances = []
        for i, (fx, fy) in enumerate(zip(feat_x, feat_y)):
            # Normalize features
            fx = F.normalize(fx, p=2, dim=1)
            fy = F.normalize(fy, p=2, dim=1)

            # Compute difference
            diff = (fx - fy) ** 2

            # Apply learned linear layer
            diff = self.lin_layers[i](diff)

            # Apply scaling
            diff = self.scaling_layers[i](diff)

            # Spatial average
            diff = diff.mean(dim=[2, 3])  # [B, 1]
            distances.append(diff)

        # Sum across layers
        total_dist = torch.cat(distances, dim=1).sum(dim=1)  # [B]
        return total_dist.mean()


def create_lpips_loss(config: dict) -> LPIPSLoss:
    """Factory function."""
    pretrained_path = config["paths"].get("pretrained_lpips", None)
    return LPIPSLoss(
        net="alex",
        version="0.1",
        pretrained_path=pretrained_path,
        normalize=True
    )