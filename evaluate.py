"""
Evaluation script for Shinobi Hiding.
Comprehensive evaluation: quality metrics, robustness, steganalysis attack.
"""
import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.full_model import create_model
from data.dataset import create_dataloaders
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.helpers import get_device, move_to_device
from utils.logger import setup_logger
from utils.metrics import (
    calculate_psnr, calculate_ssim, calculate_ber,
    calculate_bit_accuracy, calculate_message_accuracy,
    evaluate_stego_quality, evaluate_secret_reconstruction
)
from attacks.steganalysis import create_attacker, SteganalysisAttacker
from attacks.robustness import test_robustness, create_default_transforms_config


def parse_args():
    parser = argparse.ArgumentParser(description="Shinobi Hiding Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="text", choices=["text", "image"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--steganalysis", action="store_true", help="Run steganalysis attack test")
    parser.add_argument("--robustness", action="store_true", help="Run robustness test")
    parser.add_argument("--max-batches", type=int, default=50, help="Max batches for robustness test")
    return parser.parse_args()


def evaluate_text(model, test_loader, device, logger, save_figures=False, output_dir="outputs"):
    """Evaluate text steganography model."""
    model.eval()

    psnr_vals = []
    ssim_vals = []
    ber_vals = []
    bit_acc_vals = []
    msg_acc_vals = []

    with torch.no_grad():
        for batch_idx, (cover, message) in enumerate(tqdm(test_loader, desc="Evaluating")):
            cover = cover.to(device)
            message = message.to(device)

            # Generate stego
            outputs = model(cover, message, epoch=999)  # epoch=999 -> no noise
            stego = outputs["stego"]
            decoded_logits = outputs["decoded_logits"]

            # Metrics
            metrics = evaluate_stego_quality(cover, stego, message, decoded_logits)

            psnr_vals.append(metrics["psnr"])
            ssim_vals.append(metrics["ssim"])
            ber_vals.append(metrics["ber"])
            bit_acc_vals.append(metrics["bit_accuracy"])
            msg_acc_vals.append(metrics["message_accuracy"])

            # Save figures
            if save_figures and batch_idx < 5:
                from utils.helpers import save_images
                os.makedirs(output_dir, exist_ok=True)
                vis = torch.cat([cover[:4], stego[:4]], dim=0)
                save_images(vis, output_dir, f"eval_batch_{batch_idx}", nrow=4)

    results = {
        "psnr": np.mean(psnr_vals),
        "psnr_std": np.std(psnr_vals),
        "ssim": np.mean(ssim_vals),
        "ssim_std": np.std(ssim_vals),
        "ber": np.mean(ber_vals),
        "ber_std": np.std(ber_vals),
        "bit_accuracy": np.mean(bit_acc_vals),
        "bit_accuracy_std": np.std(bit_acc_vals),
        "message_accuracy": np.mean(msg_acc_vals),
        "message_accuracy_std": np.std(msg_acc_vals),
    }

    logger.info(f"Text Steganography Results:")
    logger.info(f"  PSNR: {results['psnr']:.2f} ± {results['psnr_std']:.2f} dB")
    logger.info(f"  SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    logger.info(f"  BER: {results['ber']:.2f}% ± {results['ber_std']:.2f}%")
    logger.info(f"  Bit Accuracy: {results['bit_accuracy']:.2f}% ± {results['bit_accuracy_std']:.2f}%")
    logger.info(f"  Message Accuracy: {results['message_accuracy']:.2f}% ± {results['message_accuracy_std']:.2f}%")

    return results


def evaluate_image(model, test_loader, device, logger, save_figures=False, output_dir="outputs"):
    """Evaluate image steganography model."""
    model.eval()

    cover_psnr_vals = []
    cover_ssim_vals = []
    secret_psnr_vals = []
    secret_ssim_vals = []
    secret_lpips_vals = []

    with torch.no_grad():
        for batch_idx, (cover, secret) in enumerate(tqdm(test_loader, desc="Evaluating")):
            cover = cover.to(device)
            secret = secret.to(device)

            outputs = model(cover, secret, epoch=999)
            stego = outputs["stego"]
            revealed = outputs["revealed_secret"]

            # Cover quality
            cover_psnr = calculate_psnr(cover, stego)
            cover_ssim = calculate_ssim(cover, stego)

            # Secret reconstruction
            secret_metrics = evaluate_secret_reconstruction(secret, revealed)

            cover_psnr_vals.append(cover_psnr)
            cover_ssim_vals.append(cover_ssim)
            secret_psnr_vals.append(secret_metrics["psnr"])
            secret_ssim_vals.append(secret_metrics["ssim"])
            secret_lpips_vals.append(secret_metrics["lpips"])

            if save_figures and batch_idx < 5:
                from utils.helpers import save_images
                os.makedirs(output_dir, exist_ok=True)
                vis = torch.cat([cover[:4], stego[:4], secret[:4], revealed[:4]], dim=0)
                save_images(vis, output_dir, f"eval_batch_{batch_idx}", nrow=4)

    results = {
        "cover_psnr": np.mean(cover_psnr_vals),
        "cover_psnr_std": np.std(cover_psnr_vals),
        "cover_ssim": np.mean(cover_ssim_vals),
        "cover_ssim_std": np.std(cover_ssim_vals),
        "secret_psnr": np.mean(secret_psnr_vals),
        "secret_psnr_std": np.std(secret_psnr_vals),
        "secret_ssim": np.mean(secret_ssim_vals),
        "secret_ssim_std": np.std(secret_ssim_vals),
        "secret_lpips": np.mean(secret_lpips_vals),
        "secret_lpips_std": np.std(secret_lpips_vals),
    }

    logger.info(f"Image Steganography Results:")
    logger.info(f"  Cover PSNR: {results['cover_psnr']:.2f} ± {results['cover_psnr_std']:.2f} dB")
    logger.info(f"  Cover SSIM: {results['cover_ssim']:.4f} ± {results['cover_ssim_std']:.4f}")
    logger.info(f"  Secret PSNR: {results['secret_psnr']:.2f} ± {results['secret_psnr_std']:.2f} dB")
    logger.info(f"  Secret SSIM: {results['secret_ssim']:.4f} ± {results['secret_ssim_std']:.4f}")
    logger.info(f"  Secret LPIPS: {results['secret_lpips']:.4f} ± {results['secret_lpips_std']:.4f}")

    return results


def run_steganalysis_test(model, test_loader, device, logger, config):
    """Run steganalysis attack test."""
    logger.info("Running steganalysis attack test...")

    # Collect cover and stego images
    cover_imgs = []
    stego_imgs = []

    model.eval()
    with torch.no_grad():
        for cover, message in test_loader:
            cover = cover.to(device)
            message = message.to(device)

            outputs = model(cover, message, epoch=999)
            stego = outputs["stego"]

            cover_imgs.append(cover.cpu())
            stego_imgs.append(stego.cpu())

            if len(cover_imgs) * cover.size(0) >= 1000:  # Limit for speed
                break

    cover_imgs = torch.cat(cover_imgs, dim=0)[:1000]
    stego_imgs = torch.cat(stego_imgs, dim=0)[:1000]

    # Test with SRNet
    paths = config["paths"]
    attacker = SteganalysisAttacker(
        model_name="srnet",
        weights_path=paths.get("pretrained_srnet"),
        device=device
    )

    results = attacker.evaluate(cover_imgs, stego_imgs)

    logger.info(f"Steganalysis Results (SRNet):")
    logger.info(f"  AUC: {results['auc']:.4f}")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Cover Accuracy: {results['cover_accuracy']:.4f}")
    logger.info(f"  Stego Accuracy: {results['stego_accuracy']:.4f}")

    # Also test XuNet if available
    if paths.get("pretrained_xunet") and os.path.exists(paths["pretrained_xunet"]):
        attacker_xu = SteganalysisAttacker(
            model_name="xunet",
            weights_path=paths["pretrained_xunet"],
            device=device
        )
        results_xu = attacker_xu.evaluate(cover_imgs, stego_imgs)
        logger.info(f"Steganalysis Results (XuNet):")
        logger.info(f"  AUC: {results_xu['auc']:.4f}")
        logger.info(f"  Accuracy: {results_xu['accuracy']:.4f}")
        results["xunet"] = results_xu

    return results


def run_robustness_test(model, test_loader, device, logger):
    """Run robustness test."""
    logger.info("Running robustness test...")

    transforms_config = create_default_transforms_config()
    results = test_robustness(
        model, test_loader, transforms_config, device,
        max_batches=10
    )

    logger.info("Robustness Results:")
    for name, metrics in results.items():
        logger.info(f"  {name}: BER={metrics['ber']:.2f}%, BitAcc={metrics['bit_accuracy']:.2f}%, MsgAcc={metrics['message_accuracy']:.2f}%")

    return results


def main():
    args = parse_args()

    # Setup
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("shinobi_eval")

    # Load config
    config = load_config(args.config)
    config["mode"] = args.mode

    # Load model
    model = create_model(config, mode=args.mode)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Mode: {args.mode}, Device: {device}")

    # Create test loader
    _, _, test_loader = create_dataloaders(config, mode=args.mode)
    logger.info(f"Test batches: {len(test_loader)}")

    # Run evaluations
    all_results = {"config": config}

    if args.mode == "text":
        all_results["quality"] = evaluate_text(model, test_loader, device, logger, args.save_figures)
    else:
        all_results["quality"] = evaluate_image(model, test_loader, device, logger, args.save_figures)

    if args.steganalysis:
        all_results["steganalysis"] = run_steganalysis_test(model, test_loader, device, logger, config)

    if args.robustness:
        all_results["robustness"] = run_robustness_test(model, test_loader, device, logger)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()