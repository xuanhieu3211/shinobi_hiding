"""
Training script for Shinobi Hiding.
Supports both text and image steganography branches.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.full_model import create_model
from losses.total_loss import create_loss
from data.dataset import create_dataloaders
from utils.config import load_config, get_default_config
from utils.checkpoint import CheckpointManager
from utils.logger import setup_logger, TensorBoardLogger, MetricLogger
from utils.helpers import (
    set_seed, get_device, move_to_device, setup_amp,
    setup_channels_last, compile_model, create_optimizer, create_scheduler,
    count_parameters
)
from utils.metrics import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Shinobi Hiding Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="text", choices=["text", "image"])
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    config: dict,
    device: torch.device,
    metric_logger: MetricLogger,
    use_amp: bool = True,
    channels_last: bool = False
):
    """Train for one epoch."""
    model.train()

    loss_weights = config["loss"]
    grad_accum = config["training"]["gradient_accumulation"]
    grad_clip = config["training"]["grad_clip"]

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        batch = move_to_device(batch, device)

        if config["data"]["dataset"] == "celeba":
            cover, message = batch
        else:
            cover, message = batch

        # Channels last
        if channels_last:
            cover = cover.to(memory_format=torch.channels_last)
            if config["mode"] == "image":
                message = message.to(memory_format=torch.channels_last)

        # Forward with autocast
        with autocast(enabled=use_amp):
            if config["mode"] == "text":
                # Text steganography
                outputs = model(cover, message, epoch=epoch)
                stego = outputs["stego"]
                noised_stego = outputs["noised_stego"]
                decoded_logits = outputs["decoded_logits"]

                # Discriminator forward
                disc_real = model.discriminate(cover)
                disc_fake = model.discriminate(stego.detach())

                # Generator loss
                losses = loss_fn(
                    cover, stego, noised_stego, decoded_logits, message,
                    disc_real=disc_real, disc_fake=disc_fake
                )
                gen_loss = losses["total_gen"]

                # Discriminator loss
                disc_loss = losses["disc_total"]

            else:
                # Image steganography
                outputs = model(cover, message, epoch=epoch)
                stego = outputs["stego"]
                noised_stego = outputs["noised_stego"]
                revealed = outputs["revealed_secret"]

                disc_real = model.discriminate(cover)
                disc_fake = model.discriminate(stego.detach())

                losses = loss_fn(
                    cover, stego, noised_stego, message, revealed,
                    disc_real=disc_real, disc_fake=disc_fake
                )
                gen_loss = losses["total_gen"]
                disc_loss = losses["disc_total"]

        # Backward - Generator
        gen_loss = gen_loss / grad_accum
        if use_amp:
            scaler.scale(gen_loss).backward()
        else:
            gen_loss.backward()

        # Backward - Discriminator (separate)
        disc_loss = disc_loss / grad_accum
        if use_amp:
            scaler.scale(disc_loss).backward()
        else:
            disc_loss.backward()

        # Step optimizers
        if (batch_idx + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer_g)
                scaler.unscale_(optimizer_d)

            # Gradient clipping
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)

            if use_amp:
                scaler.step(optimizer_g)
                scaler.step(optimizer_d)
                scaler.update()
            else:
                optimizer_g.step()
                optimizer_d.step()

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

        # Log metrics
        log_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        metric_logger.update(log_dict)

        # Progress
        if batch_idx % 50 == 0:
            lr = optimizer_g.param_groups[0]['lr']
            print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Gen: {losses.get('total_gen', 0):.4f} "
                  f"Disc: {losses.get('disc_total', 0):.4f} "
                  f"LR: {lr:.2e}")

    metric_logger.log_epoch(epoch, "Train")


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader,
    epoch: int,
    config: dict,
    device: torch.device,
    metric_logger: MetricLogger,
    use_amp: bool = True,
    save_images: bool = False,
    output_dir: str = "outputs"
):
    """Validate model."""
    model.eval()

    from utils.metrics import calculate_psnr, calculate_ssim, calculate_ber, calculate_bit_accuracy, calculate_message_accuracy

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ber_meter = AverageMeter()
    bit_acc_meter = AverageMeter()
    msg_acc_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = move_to_device(batch, device)

            if config["data"]["dataset"] == "celeba":
                cover, message = batch
            else:
                cover, message = batch

            if config["mode"] == "text":
                outputs = model(cover, message, epoch=epoch)
                stego = outputs["stego"]
                decoded_logits = outputs["decoded_logits"]

                # Metrics
                psnr = calculate_psnr(cover, stego)
                ssim = calculate_ssim(cover, stego)
                ber = calculate_ber(decoded_logits, message)
                bit_acc = calculate_bit_accuracy(decoded_logits, message)
                msg_acc = calculate_message_accuracy(decoded_logits, message)

            else:
                outputs = model(cover, message, epoch=epoch)
                stego = outputs["stego"]
                revealed = outputs["revealed_secret"]

                psnr = calculate_psnr(cover, stego)
                ssim = calculate_ssim(cover, stego)
                # For image branch, evaluate secret reconstruction
                from utils.metrics import evaluate_secret_reconstruction
                secret_metrics = evaluate_secret_reconstruction(message, revealed)
                ber = 100 - secret_metrics["psnr"]  # Use PSNR as proxy
                bit_acc = secret_metrics["psnr"]
                msg_acc = secret_metrics["ssim"] * 100

            psnr_meter.update(psnr, cover.size(0))
            ssim_meter.update(ssim, cover.size(0))
            ber_meter.update(ber, cover.size(0))
            bit_acc_meter.update(bit_acc, cover.size(0))
            msg_acc_meter.update(msg_acc, cover.size(0))

            # Save visualization images
            if save_images and batch_idx == 0:
                from utils.helpers import save_images, make_grid
                os.makedirs(output_dir, exist_ok=True)

                if config["mode"] == "text":
                    vis_imgs = torch.cat([cover[:8], stego[:8]], dim=0)
                else:
                    vis_imgs = torch.cat([cover[:4], stego[:4], message[:4], revealed[:4]], dim=0)

                save_images(vis_imgs, output_dir, f"val_epoch_{epoch:04d}", nrow=8)

    metrics = {
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
        "ber": ber_meter.avg,
        "bit_accuracy": bit_acc_meter.avg,
        "message_accuracy": msg_acc_meter.avg,
    }

    metric_logger.update(metrics, step=epoch)
    metric_logger.log_epoch(epoch, "Val")

    return metrics


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)
    config["mode"] = args.mode

    # Override config with CLI args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Setup logging
    log_dir = config["paths"]["logs"]
    logger = setup_logger("shinobi", log_dir)
    tb_logger = TensorBoardLogger(log_dir)
    metric_logger = MetricLogger(logger, tb_logger)

    # Log config
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"LR: {config['training']['lr']}")

    # Create model
    model = create_model(config, mode=args.mode)
    model = model.to(device)

    # VRAM optimizations
    if not args.no_channels_last:
        model = setup_channels_last(model)
    use_compile = config["training"].get("compile_model", False) and not args.no_compile and hasattr(torch, 'compile') and device.type == 'cuda'
    if use_compile:
        model = compile_model(model)
    else:
        logger.info("torch.compile() disabled (CPU mode or config)")

    # Count parameters
    total, trainable = count_parameters(model)
    logger.info(f"Model parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    # Create loss
    loss_fn = create_loss(config, mode=args.mode)
    loss_fn = loss_fn.to(device)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(config, mode=args.mode)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create optimizers
    # Separate optimizers for generator and discriminator
    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    disc_params = list(model.discriminator.parameters())

    optimizer_g = optim.AdamW(
        gen_params,
        lr=config["training"]["lr"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["weight_decay"]
    )
    optimizer_d = optim.AdamW(
        disc_params,
        lr=config["training"]["lr_discriminator"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["weight_decay"]
    )

    # Schedulers
    scheduler_g = create_scheduler(optimizer_g, config["training"], config["training"]["epochs"])
    scheduler_d = create_scheduler(optimizer_d, config["training"], config["training"]["epochs"])

    # AMP
    use_amp = config["training"]["mixed_precision"] and not args.no_amp and device.type == 'cuda'
    scaler = setup_amp(model, optimizer_g, use_amp)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        config["paths"]["checkpoints"],
        max_keep=5,
        save_best=True,
        best_metric=config["validation"]["metric"],
        best_mode="min" if config["validation"]["metric"] in ["ber", "loss"] else "max"
    )

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer_g, scheduler_g, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start = time.time()

        # Train
        train_one_epoch(
            model, loss_fn, train_loader,
            optimizer_g, optimizer_d, scaler,
            epoch, config, device, metric_logger,
            use_amp, not args.no_channels_last
        )

        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()

        # Validate
        if (epoch + 1) % config["validation"]["interval"] == 0:
            val_metrics = validate(
                model, loss_fn, val_loader,
                epoch, config, device, metric_logger,
                use_amp, config["validation"]["save_images"],
                config["paths"]["outputs"]
            )

            # Save checkpoint
            checkpoint_manager.save(
                model, optimizer_g, scheduler_g,
                epoch, val_metrics,
                extra_state={"optimizer_d": optimizer_d.state_dict(), "scheduler_d": scheduler_d.state_dict()}
            )

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

    # Final save
    final_path = Path(config["paths"]["checkpoints"]) / "final.pth"
    torch.save({
        'epoch': config["training"]["epochs"] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'config': config,
    }, final_path)
    logger.info(f"Training completed. Final model saved to {final_path}")

    tb_logger.close()


if __name__ == "__main__":
    main()