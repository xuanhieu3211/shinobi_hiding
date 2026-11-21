"""
Shinobi Hiding - Gradio Web Interface.
Provides easy-to-use GUI for encoding/decoding text and images.
"""
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.full_model import TextStegoModel, ImageStegoModel, create_model
from utils.config import load_config, get_default_config
from utils.checkpoint import load_checkpoint
from utils.helpers import get_device, tensor_to_pil
from data.transforms import denormalize


class ShinobiApp:
    """Main application class for Shinobi Hiding."""

    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None, mode: str = "text"):
        self.mode = mode
        self.device = get_device()

        # Load config
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = get_default_config(mode)

        # Create model
        self.model = create_model(self.config, mode=mode)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model, device=self.device)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded. Using random weights.")

    def encode_text(self, cover_image: Image.Image, text: str) -> Tuple[Image.Image, str]:
        """
        Encode text into cover image.
        Returns: (stego_image, info_text)
        """
        try:
            # Preprocess cover image
            cover_tensor = self._preprocess_image(cover_image)
            cover_tensor = cover_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

            # Convert text to bits
            message = self._text_to_bits(text, self.config["model"]["message_length"])
            message = message.unsqueeze(0).to(self.device)  # [1, msg_len]

            # Encode
            with torch.no_grad():
                stego = self.model.encoder(cover_tensor, message)

            # Convert to PIL
            stego_pil = tensor_to_pil(stego.squeeze(0))

            # Verify by decoding
            with torch.no_grad():
                logits = self.model.decoder(stego)
                decoded_bits = (torch.sigmoid(logits) > 0.5).float()
                decoded_text = self._bits_to_text(decoded_bits.squeeze(0))

            info = f"✓ Encoded successfully!\nOriginal: {text}\nDecoded: {decoded_text}"
            return stego_pil, info

        except Exception as e:
            return cover_image, f"✗ Error: {str(e)}"

    def decode_text(self, stego_image: Image.Image) -> str:
        """Decode text from stego image."""
        try:
            stego_tensor = self._preprocess_image(stego_image)
            stego_tensor = stego_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model.decoder(stego_tensor)
                decoded_bits = (torch.sigmoid(logits) > 0.5).float()
                decoded_text = self._bits_to_text(decoded_bits.squeeze(0))

            return f"Decoded text: {decoded_text}"

        except Exception as e:
            return f"✗ Error: {str(e)}"

    def encode_image(self, cover_image: Image.Image, secret_image: Image.Image) -> Tuple[Image.Image, Image.Image, str]:
        """
        Encode secret image into cover image (Image Steganography).
        Returns: (stego_image, revealed_secret, info_text)
        """
        try:
            # Preprocess images
            cover_tensor = self._preprocess_image(cover_image, size=self.config["model"]["cover_size"])
            secret_tensor = self._preprocess_image(secret_image, size=self.config["model"]["secret_size"])

            cover_tensor = cover_tensor.unsqueeze(0).to(self.device)
            secret_tensor = secret_tensor.unsqueeze(0).to(self.device)

            # Encode
            with torch.no_grad():
                stego = self.model.encoder(cover_tensor, secret_tensor)

            # Reveal secret
            with torch.no_grad():
                revealed = self.model.decoder(stego)

            stego_pil = tensor_to_pil(stego.squeeze(0))
            revealed_pil = tensor_to_pil(revealed.squeeze(0))

            return stego_pil, revealed_pil, "✓ Image encoded successfully!"

        except Exception as e:
            return cover_image, secret_image, f"✗ Error: {str(e)}"

    def _preprocess_image(self, pil_img: Image.Image, size: int = 256) -> torch.Tensor:
        """Preprocess PIL image to tensor in [-1, 1]."""
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((size, size), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0  # [0, 1]
        tensor = tensor.permute(2, 0, 1)  # [C, H, W]
        return tensor * 2 - 1  # [-1, 1]

    def _text_to_bits(self, text: str, length: int) -> torch.Tensor:
        """Convert text to bit tensor."""
        byte_data = text.encode('utf-8')
        bits = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        if len(bits) >= length:
            bits = bits[:length]
        else:
            bits = bits + [0] * (length - len(bits))

        return torch.tensor(bits, dtype=torch.float32)

    def _bits_to_text(self, bits: torch.Tensor) -> str:
        """Convert bit tensor to text."""
        bits = bits.cpu().numpy().astype(int)
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_val = 0
                for j in range(8):
                    byte_val = (byte_val << 1) | bits[i + j]
                if byte_val == 0:
                    break
                byte_array.append(byte_val)
        try:
            return byte_array.decode('utf-8', errors='ignore')
        except:
            return ""


def create_app(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    mode: str = "text"
) -> gr.Blocks:
    """Create Gradio interface."""

    app = ShinobiApp(config_path, checkpoint_path, mode)

    with gr.Blocks(title="Shinobi Hiding", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🥷 Shinobi Hiding - Deep Steganography
        **Neural network-based steganography for hiding text/images in images.**

        - **Text Steganography**: Hide text messages (100-256 bits) in 256×256 images
        - **Image Steganography**: Hide 128×128 secret images in 256×256 cover images
        - Based on U-Net + Adversarial + Perceptual Loss + Curriculum Noise
        """)

        with gr.Tabs():
            # Text Steganography Tab
            with gr.TabItem("📝 Text Steganography"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Encode Text")
                        cover_input = gr.Image(type="pil", label="Cover Image (256×256)")
                        text_input = gr.Textbox(
                            label="Secret Text",
                            placeholder="Enter text to hide (max ~32 chars for 256 bits)...",
                            lines=2
                        )
                        encode_btn = gr.Button("🔒 Encode", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Result")
                        stego_output = gr.Image(type="pil", label="Stego Image")
                        encode_info = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Decode Text")
                        stego_input = gr.Image(type="pil", label="Stego Image")
                        decode_btn = gr.Button("🔓 Decode", variant="secondary")

                    with gr.Column():
                        decode_output = gr.Textbox(label="Decoded Text", interactive=False)

                # Wire up events
                encode_btn.click(
                    fn=app.encode_text,
                    inputs=[cover_input, text_input],
                    outputs=[stego_output, encode_info]
                )

                decode_btn.click(
                    fn=app.decode_text,
                    inputs=[stego_input],
                    outputs=[decode_output]
                )

            # Image Steganography Tab
            with gr.TabItem("🖼️ Image Steganography"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Encode Image")
                        cover_img = gr.Image(type="pil", label="Cover Image (256×256)")
                        secret_img = gr.Image(type="pil", label="Secret Image (128×128)")
                        encode_img_btn = gr.Button("🔒 Encode Image", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Results")
                        stego_img_out = gr.Image(type="pil", label="Stego Image")
                        revealed_img_out = gr.Image(type="pil", label="Revealed Secret")
                        encode_img_info = gr.Textbox(label="Status", interactive=False)

                encode_img_btn.click(
                    fn=app.encode_image,
                    inputs=[cover_img, secret_img],
                    outputs=[stego_img_out, revealed_img_out, encode_img_info]
                )

            # Info Tab
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                ## Shinobi Hiding - Technical Details

                ### Architecture
                - **Encoder**: U-Net (5 levels) with Residual Blocks + Skip Connections
                - **Text Decoder**: 1×1 Conv + Flatten + FC (preserves spatial correlation)
                - **Image Decoder**: Symmetric U-Net
                - **Discriminator**: PatchGAN (3-4 layers) with SpectralNorm

                ### Losses
                - **Pixel Loss**: MSE/L1 between cover and stego
                - **Perceptual Loss**: LPIPS (AlexNet backbone)
                - **Adversarial Loss**: LSGAN
                - **Message Loss**: BCE per bit (text) / L1+LPIPS (image)

                ### Curriculum Noise (60 epochs)
                1. **Phase 1 (0-15)**: Identity only
                2. **Phase 2 (15-30)**: JPEG90, Gaussian(0.01)
                3. **Phase 3 (30-45)**: JPEG75, Dropout(0.1), Crop(0.9)
                4. **Phase 4 (45-60)**: All noise layers

                ### VRAM Optimization (8GB)
                - Mixed Precision (FP16 AMP)
                - Gradient Accumulation (4-8 steps)
                - Channels-last memory format
                - torch.compile()

                ### Evaluation
                - PSNR > 40 dB, SSIM > 0.98, LPIPS < 0.05
                - BER ≈ 0%, Message Accuracy ≈ 100%
                - Steganalysis AUC ≈ 0.5 (random guess)
                """)

    return demo


def launch_app(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    mode: str = "text",
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False
):
    """Launch Gradio app."""
    demo = create_app(config_path, checkpoint_path, mode)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, default="text", choices=["text", "image"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    launch_app(args.config, args.checkpoint, args.mode, args.host, args.port, args.share)