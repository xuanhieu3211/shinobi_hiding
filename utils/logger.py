"""
Logging utilities for training.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None


def setup_logger(
    name: str = "shinobi",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear existing handlers

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """
    Simple TensorBoard logger wrapper.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        self._init_writer()

    def _init_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("Warning: tensorboard not installed. Logging disabled.")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int, dataformats: str = 'CHW'):
        if self.writer:
            self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(self, tag: str, images: torch.Tensor, step: int, dataformats: str = 'NCHW'):
        if self.writer:
            self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer:
            self.writer.close()


class MetricLogger:
    """
    Tracks and logs metrics during training/validation.
    """
    def __init__(self, logger: logging.Logger, tb_logger: Optional[TensorBoardLogger] = None):
        self.logger = logger
        self.tb_logger = tb_logger
        self.metrics = {}
        self.step = 0

    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Update metrics and log to tensorboard."""
        if step is not None:
            self.step = step

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

            if self.tb_logger:
                self.tb_logger.log_scalar(key, value, self.step)

    def log_epoch(self, epoch: int, prefix: str = "Train"):
        """Log epoch summary."""
        if not self.metrics:
            return

        msg_parts = [f"{prefix} Epoch {epoch}"]
        for key, values in self.metrics.items():
            avg_val = sum(values) / len(values)
            msg_parts.append(f"{key}: {avg_val:.6f}")
            if self.tb_logger:
                self.tb_logger.log_scalar(f"{prefix}/{key}_epoch", avg_val, epoch)

        self.logger.info(" | ".join(msg_parts))
        self.metrics.clear()

    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None