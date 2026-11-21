"""
Datasets for Steganography Training.
Supports CelebA, ImageNet subsets, and custom directories.
"""
import os
import random
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import CelebA
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class StegoDataset(data.Dataset):
    """Base dataset for cover images."""
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    ):
        self.root = root
        self.image_size = image_size
        self.split = split
        self.extensions = extensions

        # Find all images
        self.image_paths = self._find_images(root)

        # Split dataset
        random.seed(seed)
        random.shuffle(self.image_paths)

        n_total = len(self.image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            self.image_paths = self.image_paths[:n_train]
        elif split == "val":
            self.image_paths = self.image_paths[n_train:n_train + n_val]
        elif split == "test":
            self.image_paths = self.image_paths[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    def _find_images(self, root: str) -> List[str]:
        """Recursively find all images in root directory."""
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(self.extensions):
                    paths.append(os.path.join(dirpath, fname))
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class TextStegoDataset(data.Dataset):
    """
    Dataset for Text Steganography.
    Returns: cover image, message bits
    """
    def __init__(
        self,
        root: str,
        message_length: int = 100,
        image_size: int = 256,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        message_source: str = "random_bits",  # "random_bits", "text_file", "ascii"
        text_corpus: Optional[str] = None,
    ):
        self.base_dataset = StegoDataset(
            root, image_size, split, train_ratio, val_ratio, seed
        )
        self.message_length = message_length
        self.message_source = message_source
        self.text_corpus = text_corpus
        self.text_lines = []

        if message_source == "text_file" and text_corpus:
            self._load_text_corpus(text_corpus)

        # ASCII printable characters (32-126)
        self.ascii_chars = [chr(i) for i in range(32, 127)]

    def _load_text_corpus(self, path: str):
        """Load text corpus for message generation."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text_lines = [line.strip() for line in f if line.strip()]

    def _generate_message(self) -> torch.Tensor:
        """Generate random message bits."""
        if self.message_source == "random_bits":
            return torch.randint(0, 2, (self.message_length,), dtype=torch.float32)

        elif self.message_source == "text_file" and self.text_lines:
            # Sample random text line
            text = random.choice(self.text_lines)
            return self._text_to_bits(text)

        elif self.message_source == "ascii":
            # Random ASCII string
            length = self.message_length // 8
            text = ''.join(random.choices(self.ascii_chars, k=length))
            return self._text_to_bits(text)

        # Fallback
        return torch.randint(0, 2, (self.message_length,), dtype=torch.float32)

    def _text_to_bits(self, text: str) -> torch.Tensor:
        """Convert text string to bit tensor."""
        # Encode as UTF-8 bytes
        byte_data = text.encode('utf-8')
        bits = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        # Truncate or pad to message_length
        if len(bits) >= self.message_length:
            bits = bits[:self.message_length]
        else:
            bits = bits + [0] * (self.message_length - len(bits))

        return torch.tensor(bits, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cover = self.base_dataset[idx]
        message = self._generate_message()
        return cover, message


class ImageStegoDataset(data.Dataset):
    """
    Dataset for Image Steganography.
    Returns: cover image, secret image
    """
    def __init__(
        self,
        root: str,
        cover_size: int = 256,
        secret_size: int = 128,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.base_dataset = StegoDataset(
            root, cover_size, split, train_ratio, val_ratio, seed
        )
        self.secret_size = secret_size

        # Separate transform for secret (smaller)
        self.secret_transform = transforms.Compose([
            transforms.Resize((secret_size, secret_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get cover image
        cover = self.base_dataset[idx]

        # Get secret image (different random image from dataset)
        secret_idx = random.randint(0, len(self.base_dataset) - 1)
        secret_img = self.base_dataset[secret_idx]

        # Resize secret to secret_size
        # secret_img is already tensor [3, H, W] in [-1, 1]
        secret = transforms.functional.resize(
            secret_img, (self.secret_size, self.secret_size),
            interpolation=transforms.InterpolationMode.BILINEAR
        )

        return cover, secret


class CelebAStegoDataset(data.Dataset):
    """
    CelebA dataset wrapper for steganography.
    Uses torchvision CelebA dataset with custom splits.
    """
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",
        download: bool = False,
        message_length: int = 100,
        mode: str = "text"  # "text" or "image"
    ):
        self.mode = mode
        self.message_length = message_length

        # Map split names
        split_map = {"train": "train", "val": "valid", "test": "test"}
        celeba_split = split_map.get(split, "train")

        self.dataset = CelebA(
            root=root,
            split=celeba_split,
            target_type="attr",
            transform=transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]),
            download=download
        )

        if mode == "image":
            self.secret_size = 128
            self.secret_transform = transforms.Compose([
                transforms.Resize((128, 128), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]  # Ignore attributes

        if self.mode == "text":
            # Random message
            message = torch.randint(0, 2, (self.message_length,), dtype=torch.float32)
            return img, message
        else:
            # Random secret image
            secret_idx = random.randint(0, len(self.dataset) - 1)
            secret_img, _ = self.dataset[secret_idx]
            secret = self.secret_transform(secret_img)
            return img, secret


def create_dataloaders(config: dict, mode: str = "text") -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Create train/val/test dataloaders from config.
    Returns: (train_loader, val_loader, test_loader)
    """
    data_config = config["data"]
    train_config = config["training"]

    if data_config["dataset"] == "celeba":
        train_dataset = CelebAStegoDataset(
            root=data_config["root"],
            image_size=data_config["image_size"],
            split="train",
            download=True,
            message_length=data_config.get("message_length", 100),
            mode=mode
        )
        val_dataset = CelebAStegoDataset(
            root=data_config["root"],
            image_size=data_config["image_size"],
            split="val",
            download=False,
            message_length=data_config.get("message_length", 100),
            mode=mode
        )
        test_dataset = CelebAStegoDataset(
            root=data_config["root"],
            image_size=data_config["image_size"],
            split="test",
            download=False,
            message_length=data_config.get("message_length", 100),
            mode=mode
        )
    else:
        # Custom directory dataset
        if mode == "text":
            train_dataset = TextStegoDataset(
                root=data_config["root"],
                message_length=data_config.get("message_length", 100),
                image_size=data_config["image_size"],
                split="train",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"],
                message_source=data_config.get("message_source", "random_bits"),
                text_corpus=data_config.get("text_corpus", None)
            )
            val_dataset = TextStegoDataset(
                root=data_config["root"],
                message_length=data_config.get("message_length", 100),
                image_size=data_config["image_size"],
                split="val",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"],
                message_source=data_config.get("message_source", "random_bits"),
                text_corpus=data_config.get("text_corpus", None)
            )
            test_dataset = TextStegoDataset(
                root=data_config["root"],
                message_length=data_config.get("message_length", 100),
                image_size=data_config["image_size"],
                split="test",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"],
                message_source=data_config.get("message_source", "random_bits"),
                text_corpus=data_config.get("text_corpus", None)
            )
        else:
            train_dataset = ImageStegoDataset(
                root=data_config["root"],
                cover_size=data_config["cover_size"],
                secret_size=data_config["secret_size"],
                split="train",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"]
            )
            val_dataset = ImageStegoDataset(
                root=data_config["root"],
                cover_size=data_config["cover_size"],
                secret_size=data_config["secret_size"],
                split="val",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"]
            )
            test_dataset = ImageStegoDataset(
                root=data_config["root"],
                cover_size=data_config["cover_size"],
                secret_size=data_config["secret_size"],
                split="test",
                train_ratio=data_config["train_split"],
                val_ratio=data_config["val_split"]
            )

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=True,
        persistent_workers=train_config["num_workers"] > 0
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=False,
        persistent_workers=train_config["num_workers"] > 0
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=False
    )

    return train_loader, val_loader, test_loader