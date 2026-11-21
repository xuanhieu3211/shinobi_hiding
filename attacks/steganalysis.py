"""
Steganalysis Attack Testing.
Tests resistance against SRNet, XuNet, Yedroudj-Net classifiers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os


class SRNet(nn.Module):
    """
    SRNet (Spatial Rich Model Network) for steganalysis.
    Based on: "SRNet: A Deep Network for Steganalysis" (CVPR 2018)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Layer 2-3 (residual blocks)
        self.layer2 = self._make_layer(64, 16, 2)
        self.layer3 = self._make_layer(16, 16, 2)

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=1),
        )

        # Layer 5-8
        self.layer5 = self._make_layer(16, 64, 2)
        self.layer6 = self._make_layer(64, 128, 2)
        self.layer7 = self._make_layer(128, 256, 2)
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 and in_channels != out_channels else 1
            layers.append(SRNetBlock(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SRNetBlock(nn.Module):
    """Residual block for SRNet."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class XuNet(nn.Module):
    """
    XuNet for steganalysis.
    Based on: "Deep Convolutional Neural Network for Steganalysis" (2016)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Kernel initialization (HPF - High Pass Filters)
        self.hpf = nn.Conv2d(3, 3, 5, padding=2, bias=False)
        self._init_hpf()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.AvgPool2d(5, stride=2, padding=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(5, stride=2, padding=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(5, stride=2, padding=2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _init_hpf(self):
        """Initialize with High Pass Filters (HPF)."""
        # Simple HPF kernels for RGB
        hpf_weights = torch.zeros(3, 3, 5, 5)
        for c in range(3):
            # Simple edge detection kernel
            hpf_weights[c, c] = torch.tensor([
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ], dtype=torch.float32) / 12.0
        self.hpf.weight.data = hpf_weights
        self.hpf.weight.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.hpf:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hpf(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SteganalysisAttacker:
    """
    Steganalysis attack tester.
    Loads pretrained steganalysis models and evaluates detection accuracy.
    """
    def __init__(
        self,
        model_name: str = "srnet",
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_classes: int = 2
    ):
        self.model_name = model_name.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        if self.model_name == "srnet":
            self.model = SRNet(num_classes)
        elif self.model_name == "xunet":
            self.model = XuNet(num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load weights
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            print(f"Warning: No weights loaded for {model_name}. Using random initialization.")

        self.model = self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path: str):
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:5]}...")
        print(f"Loaded {self.model_name} weights from {path}")

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict steganalysis labels.
        Args:
            images: [B, 3, H, W] in range [-1, 1] or [0, 1]
        Returns:
            logits: [B, 2] or probs: [B, 2]
        """
        self.model.eval()
        with torch.no_grad():
            # Ensure correct range
            if images.min() < 0:
                images = (images + 1) / 2  # [-1, 1] -> [0, 1]
            images = images.to(self.device)
            logits = self.model(images)
        return logits

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.predict(images)
        return F.softmax(logits, dim=1)

    def evaluate(
        self,
        cover_images: torch.Tensor,
        stego_images: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate steganalysis detection performance.
        Args:
            cover_images: [N, 3, H, W] cover images
            stego_images: [N, 3, H, W] stego images
        Returns:
            Dict with AUC, Accuracy, etc.
        """
        # Prepare labels
        cover_labels = torch.zeros(len(cover_images), dtype=torch.long)
        stego_labels = torch.ones(len(stego_images), dtype=torch.long)

        all_labels = torch.cat([cover_labels, stego_labels])
        all_images = torch.cat([cover_images, stego_images])

        # Predict in batches
        all_probs = []
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            probs = self.predict_proba(batch)
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs, dim=0)

        # Compute metrics
        y_true = all_labels.numpy()
        y_score = all_probs[:, 1].numpy()  # Probability of stego class
        y_pred = (y_score > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)

        # Per-class accuracy
        cover_acc = accuracy_score(y_true[:len(cover_images)], y_pred[:len(cover_images)])
        stego_acc = accuracy_score(y_true[len(cover_images):], y_pred[len(cover_images):])

        return {
            "auc": auc,
            "accuracy": acc,
            "cover_accuracy": cover_acc,
            "stego_accuracy": stego_acc,
            "model": self.model_name,
        }

    def fine_tune(
        self,
        cover_images: torch.Tensor,
        stego_images: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4
    ):
        """
        Fine-tune steganalysis model on custom stego images.
        """
        self.model.train()

        # Prepare dataset
        labels = torch.cat([
            torch.zeros(len(cover_images), dtype=torch.long),
            torch.ones(len(stego_images), dtype=torch.long)
        ])
        images = torch.cat([cover_images, stego_images])

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if batch_images.min() < 0:
                    batch_images = (batch_images + 1) / 2

                optimizer.zero_grad()
                logits = self.model(batch_images)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Fine-tune epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()


def load_srnet(weights_path: str, device: Optional[torch.device] = None) -> SRNet:
    """Load pretrained SRNet."""
    model = SRNet()
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device or 'cpu', weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model


def load_xunet(weights_path: str, device: Optional[torch.device] = None) -> XuNet:
    """Load pretrained XuNet."""
    model = XuNet()
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device or 'cpu', weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model


def create_attacker(config: dict) -> SteganalysisAttacker:
    """Factory function to create attacker from config."""
    paths = config["paths"]
    return SteganalysisAttacker(
        model_name="srnet",
        weights_path=paths.get("pretrained_srnet"),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )