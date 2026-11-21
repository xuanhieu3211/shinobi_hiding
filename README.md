# 🥷 Shinobi Hiding - Deep Steganography

**Neural network-based steganography for hiding text/images in images with perceptual quality and steganalysis resistance.**

> **Đồ án cuối kỳ - Đề tài 6: Ẩn mã thông minh dựa trên ML (Steganography bằng mạng nơ-ron)**
> - **Trọng tâm**: Ẩn chuỗi văn bản/bit (100-256 bits) trong ảnh 256×256
> - **Bổ sung**: Ẩn ảnh 128×128 trong ảnh 256×256 (Image-in-Image)
> - **Kiến trúc**: U-Net + Adversarial Loss + Perceptual Loss (LPIPS) + Curriculum Noise
> - **Sản phẩm**: Mã huấn luyện, checkpoint, đánh giá chất lượng + test tấn công steganalysis

---

## 🎯 Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| **Text Steganography** | Giấu 100-256 bits văn bản vào ảnh 256×256, BER ≈ 0% |
| **Image Steganography** | Giấu ảnh 128×128 vào ảnh 256×256 (nhánh bổ sung) |
| **Perceptual Quality** | PSNR > 40dB, SSIM > 0.98, LPIPS < 0.05 |
| **Steganalysis Resistance** | AUC ≈ 0.5 (random guess) trước SRNet/XuNet |
| **Robustness** | Khôi phục tin nhắn sau JPEG, noise, crop, rotation, resize |
| **VRAM Optimization** | Chạy được trên 8GB (RTX 5050/4050/3050) |
| **Web Demo** | Gradio interface dễ sử dụng |

---

## 🏗️ Kiến trúc mô hình

```
┌─────────────────────────────────────────────────────────────┐
                    TEXT STEGANOGRAPHY                        
├─────────────────────────────────────────────────────────────┤
  Cover (3×256×256) + Message (100 bits)                      
         │                                                     
         ▼                                                     
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      
  │   Encoder   │───▶│   Stego     │───▶│  Decoder    │      
  │  (U-Net 5   │    │  (3×256×256)│    │ (Conv+FC)   │      
  │  levels)    │    │             │    │             │      
  └─────────────┘    └─────────────┘    └─────────────┘      
         │                                      │              
         ▼                                      ▼              
  ┌─────────────┐                        ┌─────────────┐      
  │Discriminator│◀─── Adversarial ───────│  Message    │      
  │ (PatchGAN   │     Loss (LSGAN)       │  Recovery   │      
  │  Spectral)  │                        │  (BCE)      │      
  └─────────────┘                        └─────────────┘      
         │                                                     
         ▼                                                     
  ┌─────────────┐                                              
  │   LPIPS     │◀─── Perceptual Loss                         
  │  (AlexNet)  │                                              
  └─────────────┘                                              
```

### Loss Function

```
L_total = λ_pixel·L_MSE + λ_lpips·L_LPIPS + λ_adv·L_LSGAN + λ_msg·L_BCE

Text branch:  λ_pixel=1.0, λ_lpips=5.0, λ_adv=0.5, λ_msg=10.0
Image branch: λ_pixel=1.0, λ_secret=10.0, λ_lpips_cover=5.0, λ_lpips_secret=10.0, λ_adv=0.5
```

---

## 📦 Cài đặt

```bash
# 1. Clone/Download project
cd C:\Users\LamNguyenXuanHieuXD\OneDrive\LyThuyetThongTin\shinobi_hiding

# 2. Tạo environment
conda create -n shinobi python=3.10 -y
conda activate shinobi

# 3. Cài PyTorch (CUDA 12.1 cho RTX 5050/4050/3050)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Cài dependencies
pip install -r requirements.txt

# 5. (Tùy chọn) Cài diffjpeg cho JPEG differentiable
pip install diffjpeg
```

---

## 🚀 Huấn luyện

### Nhánh chính: Text Steganography (60 epochs)

```bash
# Huấn luyện từ đầu
python main.py train-text --config configs/text_stego.yaml --epochs 60

# Resume từ checkpoint
python main.py train-text --config configs/text_stego.yaml --resume checkpoints/text/epoch_0030.pth

# Tùy chỉnh hyperparameters
python main.py train-text --config configs/text_stego.yaml --batch-size 8 --lr 1e-4
```

**Checkpoints**:
- `checkpoints/text/epoch_XXXX.pth` - Mỗi epoch
- `checkpoints/text/best.pht` - Model tốt nhất (theo BER)

### Nhánh bổ sung: Image Steganography (30 epochs)

```bash
python main.py train-image --config configs/image_stego.yaml --epochs 30
```

**Checkpoints**: `checkpoints/image/`

### Theo dõi training (TensorBoard)

```bash
tensorboard --logdir logs/text
# Mở http://localhost:6006
```

---

## 📊 Đánh giá

```bash
# Đánh giá cơ bản
python main.py evaluate --config configs/text_stego.yaml --checkpoint checkpoints/text/best.pth

# Đánh giá đầy đủ (kèm steganalysis + robustness)
python main.py evaluate --config configs/text_stego.yaml --checkpoint checkpoints/text/best.pth \
    --steganalysis --robustness --save-figures --output results.json
```

### Kết quả mong đợi (Text branch)

| Metric | Target |
|--------|--------|
| PSNR | > 40 dB |
| SSIM | > 0.98 |
| LPIPS | < 0.05 |
| BER | < 1% |
| Bit Accuracy | > 99% |
| Message Accuracy | > 95% |
| Steganalysis AUC (SRNet) | ≈ 0.50 - 0.55 |

---

## 🌐 Demo Web (Gradio)

```bash
# Chạy demo text steganography
python main.py demo --config configs/text_stego.yaml --checkpoint checkpoints/text/best.pth

# Chạy demo image steganography
python main.py demo --config configs/image_stego.yaml --checkpoint checkpoints/image/best.pth --mode image
```

Mở trình duyệt tại: **http://localhost:7860**

### Tính năng demo:
- **Encode**: Chọn ảnh cover + nhập text → Tạo ảnh stego
- **Decode**: Chọn ảnh stego → Khôi phục text
- **Image Stego**: Chọn cover + secret image → Stego + Revealed secret

---

## 🔬 Steganalysis Attack Test

Project tự động test chống phát hiện với các mô hình SOTA:

| Mô hình | Mô tả | Pretrained weights |
|---------|-------|-------------------|
| **SRNet** | Spatial Rich Model Network (CVPR 2018) | `weights/srnet.pth` |
| **XuNet** | Deep CNN for Steganalysis (2016) | `weights/xunet.pth` |
| **Yedroudj-Net** | (Tùy chọn) | `weights/yedroudj.pth` |

```bash
# Tải pretrained weights (đặt vào folder weights/)
# SRNet: https://github.com/JSteg-SRNet
# XuNet: https://github.com/steganalysis-pytorch

# Chạy test
python main.py evaluate --config configs/text_stego.yaml --checkpoint checkpoints/text/best.pth --steganalysis
```

**Mục tiêu**: AUC ≈ 0.5 (tương đương đoán ngẫu nhiên)

---

## 🛡️ Robustness Test

Test khôi phục tin nhắn sau các biến đổi:

```bash
python main.py evaluate --config configs/text_stego.yaml --checkpoint checkpoints/text/best.pth --robustness
```

| Biến đổi | Parameter | Kỳ vọng BER |
|----------|-----------|-------------|
| JPEG | Quality 90/75/50 | < 5% |
| Gaussian Noise | σ=0.01/0.02 | < 3% |
| Dropout | p=0.1/0.2 | < 5% |
| Crop | scale=0.9/0.85 | < 10% |
| Resize | 0.5× / 2.0× | < 15% |
| Rotation | ±5°/±10° | < 10% |
| Brightness/Contrast | ±0.2 | < 5% |

---

## ⚙️ Cấu hình (Config YAML)

File config chính: `configs/text_stego.yaml`

```yaml
model:
  cover_size: 256
  message_length: 100        # 100 bits mặc định (tối đa 256)
  embed_dim: 64
  base_channels: 64
  encoder_depth: 5
  decoder_type: "conv_fc"    # hoặc "transformer"

training:
  epochs: 60
  batch_size: 8              # Giảm xuống 4 nếu OOM
  gradient_accumulation: 4   # Effective batch = 32
  lr: 1e-4
  mixed_precision: true
  channels_last: true
  compile_model: true

loss:
  lambda_pixel: 1.0
  lambda_lpips: 5.0
  lambda_adv: 0.5
  lambda_msg: 10.0
```

---

## 📁 Cấu trúc thư mục

```
shinobi_hiding/
├── main.py                 # Entry point
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt
├── README.md
├── configs/
│   ├── text_stego.yaml     # Config nhánh chính
│   └── image_stego.yaml    # Config nhánh bổ sung
├── models/
│   ├── encoder.py          # U-Net Encoder (shared)
│   ├── decoder_text.py     # Text Decoder (Conv+FC/Transformer)
│   ├── decoder_image.py    # Image Decoder (U-Net)
│   ├── discriminator.py    # PatchGAN SpectralNorm
│   ├── noise_layers.py     # 10 Noise Layers + Curriculum
│   └── full_model.py       # Model wrappers
├── losses/
│   ├── lpips_loss.py       # LPIPS Perceptual Loss
│   ├── adversarial_loss.py # LSGAN Adversarial Loss
│   ├── reconstruction_loss.py
│   └── total_loss.py       # Combined losses
├── data/
│   ├── dataset.py          # CelebA/Custom datasets
│   └── transforms.py       # Augmentations
├── utils/
│   ├── config.py           # Config loading
│   ├── checkpoint.py       # Checkpoint management
│   ├── metrics.py          # PSNR, SSIM, BER, etc.
│   ├── logger.py           # Logging + TensorBoard
│   └── helpers.py          # Seed, AMP, compile, etc.
├── attacks/
│   ├── steganalysis.py     # SRNet, XuNet attackers
│   └── robustness.py       # Robustness transforms
├── gui/
│   └── gradio_app.py       # Gradio web interface
├── checkpoints/            # Model checkpoints
├── logs/                   # TensorBoard logs
├── outputs/                # Visualization images
└── weights/                # Pretrained weights (lpips, srnet, xunet)
```

---

## 💡 Tips & Troubleshooting

### OOM (Out of Memory) trên 8GB VRAM
```bash
# Giảm batch size, tăng gradient accumulation
python main.py train-text --config configs/text_stego.yaml --batch-size 4

# Hoặc chỉnh config:
# batch_size: 4
# gradient_accumulation: 8
```

### BER cao (> 10%)
- Tăng `lambda_msg` (10 → 20)
- Kéo dài Phase 1-2 (clean training)
- Giảm cường độ noise sớm

### Steganalysis AUC cao (> 0.7)
- Tăng `lambda_adv` (0.5 → 1.0)
- Thêm noise mạnh sớm hơn
- Fine-tune discriminator

### Training chậm
- Bật `compile_model: true` (PyTorch 2.0+)
- Bật `channels_last: true`
- Sử dụng `num_workers: 4` + `pin_memory: true`

---

## 📝 Trích dẫn & Tham khảo

```bibtex
@inproceedings{zhu2018hidden,
  title={Hidden: Hiding data with deep networks},
  author={Zhu, Jiren and Kaplan, Russell and Johnson, Justin and Fei-Fei, Li},
  booktitle={ECCV},
  year={2018}
}

@inproceedings{boroumand2018srnet,
  title={SRNet: A deep network for steganalysis},
  author={Boroumand, Mehdi and Chen, Mo and Fridrich, Jessica},
  booktitle={CVPR},
  year={2018}
}

@article{xu2017deep,
  title={Deep convolutional neural network for steganalysis},
  author={Xu, Guanshuo and Wu, Han-Zhou and Shi, Yun-Qing},
  journal={IEEE Signal Processing Letters},
  year={2017}
}
```

---

## 👨‍💻 Tác giả

**Đồ án cuối kỳ - Đề tài 6**
- **Tên**: Shinobi Hiding
- **Kiến trúc**: U-Net + Adversarial + Perceptual + Curriculum Noise
- **Framework**: PyTorch 2.0+
- **GPU**: RTX 5050/4050/3050 (8GB VRAM)

---

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập/nghiên cứu.

---

> **Lưu ý**: Đây là đồ án học tập. Không sử dụng cho mục đích xấu. Steganography nên dùng để bảo vệ quyền sở hữu trí tuệ, watermarking, hoặc truyền tin an toàn trong môi trường được phép.