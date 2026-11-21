#!/usr/bin/env python3
"""
Shinobi Hiding - Main Entry Point
Deep Steganography with Neural Networks

Usage:
    python main.py train-text --config configs/text_stego.yaml
    python main.py train-image --config configs/image_stego.yaml
    python main.py evaluate --checkpoint checkpoints/text/best.pth --config configs/text_stego.yaml
    python main.py demo
"""
import argparse
import sys
import subprocess
from pathlib import Path


def run_train_text(args):
    """Train text steganography model."""
    cmd = [
        sys.executable, "train.py",
        "--config", args.config,
        "--mode", "text"
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.no_amp:
        cmd.append("--no-amp")
    if args.no_compile:
        cmd.append("--no-compile")
    subprocess.run(cmd, check=True)


def run_train_image(args):
    """Train image steganography model."""
    cmd = [
        sys.executable, "train.py",
        "--config", args.config,
        "--mode", "image"
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.no_amp:
        cmd.append("--no-amp")
    if args.no_compile:
        cmd.append("--no-compile")
    subprocess.run(cmd, check=True)


def run_evaluate(args):
    """Evaluate model."""
    cmd = [
        sys.executable, "evaluate.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--mode", args.mode,
        "--output", args.output
    ]
    if args.save_figures:
        cmd.append("--save-figures")
    if args.steganalysis:
        cmd.append("--steganalysis")
    if args.robustness:
        cmd.append("--robustness")
    subprocess.run(cmd, check=True)


def run_demo(args):
    """Launch Gradio demo."""
    from gui.gradio_app import launch_app
    launch_app(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        host=args.host,
        port=args.port,
        share=args.share
    )


def main():
    parser = argparse.ArgumentParser(
        description="Shinobi Hiding - Deep Steganography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train text steganography (main branch)
  python main.py train-text --config configs/text_stego.yaml --epochs 60

  # Train image steganography (supplementary branch)
  python main.py train-image --config configs/image_stego.yaml --epochs 30

  # Evaluate with steganalysis and robustness tests
  python main.py evaluate --checkpoint checkpoints/text/best.pth --config configs/text_stego.yaml --steganalysis --robustness

  # Launch web demo
  python main.py demo --checkpoint checkpoints/text/best.pth --config configs/text_stego.yaml
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train-text
    parser_train_text = subparsers.add_parser("train-text", help="Train text steganography")
    parser_train_text.add_argument("--config", type=str, default="configs/text_stego.yaml")
    parser_train_text.add_argument("--resume", type=str, default=None)
    parser_train_text.add_argument("--epochs", type=int, default=None)
    parser_train_text.add_argument("--batch-size", type=int, default=None)
    parser_train_text.add_argument("--lr", type=float, default=None)
    parser_train_text.add_argument("--no-amp", action="store_true")
    parser_train_text.add_argument("--no-compile", action="store_true")

    # train-image
    parser_train_image = subparsers.add_parser("train-image", help="Train image steganography")
    parser_train_image.add_argument("--config", type=str, default="configs/image_stego.yaml")
    parser_train_image.add_argument("--resume", type=str, default=None)
    parser_train_image.add_argument("--epochs", type=int, default=None)
    parser_train_image.add_argument("--batch-size", type=int, default=None)
    parser_train_image.add_argument("--lr", type=float, default=None)
    parser_train_image.add_argument("--no-amp", action="store_true")
    parser_train_image.add_argument("--no-compile", action="store_true")

    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model")
    parser_eval.add_argument("--config", type=str, required=True)
    parser_eval.add_argument("--checkpoint", type=str, required=True)
    parser_eval.add_argument("--mode", type=str, default="text", choices=["text", "image"])
    parser_eval.add_argument("--output", type=str, default="evaluation_results.json")
    parser_eval.add_argument("--save-figures", action="store_true")
    parser_eval.add_argument("--steganalysis", action="store_true")
    parser_eval.add_argument("--robustness", action="store_true")

    # demo
    parser_demo = subparsers.add_parser("demo", help="Launch Gradio web demo")
    parser_demo.add_argument("--config", type=str, default="configs/text_stego.yaml")
    parser_demo.add_argument("--checkpoint", type=str, default=None)
    parser_demo.add_argument("--mode", type=str, default="text", choices=["text", "image"])
    parser_demo.add_argument("--host", type=str, default="0.0.0.0")
    parser_demo.add_argument("--port", type=int, default=7860)
    parser_demo.add_argument("--share", action="store_true")

    args = parser.parse_args()

    if args.command == "train-text":
        run_train_text(args)
    elif args.command == "train-image":
        run_train_image(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()