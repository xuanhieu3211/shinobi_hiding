"""
Configuration loading and saving utilities.
"""
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Supports inheritance via '_base_' key.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if '_base_' in config:
        base_paths = config.pop('_base_')
        if isinstance(base_paths, str):
            base_paths = [base_paths]

        base_config = {}
        for base_path in base_paths:
            # Resolve relative to current config file
            base_full_path = Path(config_path).parent / base_path
            base_cfg = load_config(str(base_full_path))
            base_config = _merge_configs(base_config, base_cfg)

        # Merge with current config (current overrides base)
        config = _merge_configs(base_config, config)

    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configs. override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config(mode: str = "text") -> Dict[str, Any]:
    """Get default configuration for text or image steganography."""
    if mode == "text":
        return {
            "model": {
                "cover_size": 256,
                "message_length": 100,
                "max_message_length": 256,
                "embed_dim": 64,
                "base_channels": 64,
                "encoder_depth": 5,
                "use_residual_blocks": True,
                "decoder_type": "conv_fc",
                "transformer_layers": 2,
                "transformer_heads": 4,
            },
            "discriminator": {
                "type": "patchgan_spectral",
                "input_channels": 3,
                "base_channels": 64,
                "n_layers": 3,
                "patch_size": 70,
            },
            "noise": {
                "layers": [
                    {"name": "identity", "params": {}},
                    {"name": "jpeg", "params": {"quality_min": 75, "quality_max": 90, "differentiable": True}},
                    {"name": "gaussian", "params": {"std_min": 0.01, "std_max": 0.02}},
                    {"name": "dropout", "params": {"p_min": 0.1, "p_max": 0.2}},
                    {"name": "crop", "params": {"scale_min": 0.85, "scale_max": 1.0}},
                    {"name": "resize", "params": {"scale_min": 0.5, "scale_max": 2.0}},
                    {"name": "rotation", "params": {"degrees": 10}},
                    {"name": "brightness_contrast", "params": {"brightness": 0.2, "contrast": 0.2}},
                    {"name": "salt_pepper", "params": {"prob": 0.01}},
                    {"name": "blur", "params": {"kernel_size": 3}},
                ],
                "curriculum": [
                    {"name": "phase1_clean", "epochs": [0, 15], "active_layers": ["identity"]},
                    {"name": "phase2_light", "epochs": [15, 30], "active_layers": ["identity", "jpeg", "gaussian"], "jpeg_quality": 90, "gaussian_std": 0.01},
                    {"name": "phase3_medium", "epochs": [30, 45], "active_layers": ["identity", "jpeg", "gaussian", "dropout", "crop"], "jpeg_quality": 75, "gaussian_std": 0.01, "dropout_p": 0.1, "crop_scale": 0.9},
                    {"name": "phase4_heavy", "epochs": [45, 60], "active_layers": ["identity", "jpeg", "gaussian", "dropout", "crop", "resize", "rotation", "brightness_contrast"], "jpeg_quality": 75, "gaussian_std": 0.02, "dropout_p": 0.2, "crop_scale": 0.85},
                ],
            },
            "loss": {
                "lambda_pixel": 1.0,
                "lambda_lpips": 5.0,
                "lambda_adv": 0.5,
                "lambda_msg": 10.0,
            },
            "training": {
                "epochs": 60,
                "batch_size": 8,
                "gradient_accumulation": 4,
                "lr": 1e-4,
                "lr_discriminator": 1e-4,
                "beta1": 0.5,
                "beta2": 0.999,
                "weight_decay": 1e-5,
                "scheduler": "cosine_warmup",
                "warmup_epochs": 5,
                "grad_clip": 1.0,
                "mixed_precision": True,
                "channels_last": True,
                "compile_model": True,
                "num_workers": 4,
                "pin_memory": True,
                "seed": 42,
            },
            "validation": {
                "interval": 1,
                "save_best": True,
                "metric": "ber",
                "save_images": True,
                "num_vis_samples": 8,
            },
            "data": {
                "dataset": "celeba",
                "root": "data/celeba",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "image_size": 256,
                "message_source": "random_bits",
                "text_corpus": "data/text_corpus.txt",
            },
            "paths": {
                "checkpoints": "checkpoints/text",
                "logs": "logs/text",
                "outputs": "outputs/text",
                "pretrained_lpips": "weights/lpips_alexnet.pth",
                "pretrained_srnet": "weights/srnet.pth",
                "pretrained_xunet": "weights/xunet.pth",
            },
        }
    else:
        # Image stego config
        base = get_default_config("text")
        base["model"]["secret_size"] = 128
        base["model"]["secret_channels"] = 3
        base["model"]["decoder_type"] = "unet"
        base["training"]["epochs"] = 30
        base["training"]["batch_size"] = 4
        base["training"]["gradient_accumulation"] = 8
        base["loss"] = {
            "lambda_pixel": 1.0,
            "lambda_secret": 10.0,
            "lambda_lpips_cover": 5.0,
            "lambda_lpips_secret": 10.0,
            "lambda_adv": 0.5,
        }
        base["validation"]["metric"] = "psnr_secret"
        base["paths"]["checkpoints"] = "checkpoints/image"
        base["paths"]["logs"] = "logs/image"
        base["paths"]["outputs"] = "outputs/image"
        return base