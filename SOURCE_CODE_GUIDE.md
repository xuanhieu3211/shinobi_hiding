# Shinobi Hiding - Source Code Guide for Report Writers

> **Mục đích**: Giúp 4 thành viên nhóm hiểu nhanh từng file làm gì, để viết báo cáo **khớp 100% với code thực tế**.
> **Lưu ý**: **KHÔNG CẦN CHẠY CODE** — chỉ đọc hiểu kiến trúc, tham số, luồng dữ liệu.

---

## 📂 CẤU TRÚC THƯ MỤC

```
shinobi_hiding/
├── main.py                    # Entry point CLI
├── train.py                   # Training loop chính
├── evaluate.py                # Đánh giá + Steganalysis + Robustness
├── configs/
│   ├── text_stego.yaml        # Config nhánh chính (Text Steganography)
│   └── image_stego.yaml       # Config nhánh phụ (Image-in-Image)
├── models/                    # Kiến trúc mạng
├── losses/                    # Hàm mất mát
├── data/                      # Dataset & DataLoader
├── attacks/                   # Steganalysis & Robustness test
├── utils/                     # Helpers, config, checkpoint, logger, metrics
└── gui/                       # Gradio demo
```

---

## 🎯 FILE QUAN TRỌNG THEO PHÂN CÔNG

### 👤 Người 3 — Phần IV: Cơ sở lý thuyết & Kiến trúc
**Đọc 5 file này (ưu tiên theo thứ tự):**

| File | Chức năng | Điểm cần nắm cho báo cáo |
|------|-----------|--------------------------|
| `models/encoder.py` | **U-Net Encoder 5 cấp** + Residual Blocks + **Message Broadcaster** (mở rộng bit message thành 30 kênh feature map) | - Số tầng: 5 (downsample 256→128→64→32→16→8)<br>- Base channels: 64<br>- Residual blocks ở mỗi tầng<br>- Broadcaster: `message [B, 100] → [B, 30, 16, 16]` concat với bottleneck |
| `models/decoder_text.py` | **Text Decoder**: 1×1 Conv → Flatten → FC → logits [B, 100-256] | - **KHÔNG dùng GlobalAvgPool** (giữ thông tin không gian)<br>- 1×1 Conv giảm kênh → Flatten → Linear<br>- Output: logits per bit (BCE loss) |
| `models/noise_layers.py` | **10 Noise Layers** + **Curriculum Noise** (4 giai đoạn 60 epochs) | - 10 layers: Identity, JPEG(STE), Gaussian, Dropout, Crop, Rotation, Resize, BrightnessContrast, SaltPepper, BlurPool<br>- Curriculum: Phase1(0-15): Identity; Phase2(15-30): JPEG90+Gaussian; Phase3(30-45): +Dropout+Crop; Phase4(45-60): +Resize+Rotation+Brightness<br>- JPEG dùng **Straight-Through Estimator (STE)** cho gradient |
| `losses/total_loss.py` | **Tổng hợp Loss nhánh Text**: L = λ_pixel·L_MSE + λ_lpips·L_LPIPS + λ_msg·L_BCE + λ_adv·L_LSGAN | - λ_pixel=1.0, λ_lpips=5.0, λ_msg=10.0, λ_adv=0.5 (xem config)<br>- Pixel: MSE cover vs stego<br>- LPIPS: AlexNet frozen<br>- Message: BCE per bit trên noised_stego<br>- Adv: LSGAN generator/discriminator |
| `configs/text_stego.yaml` | **Toàn bộ hyperparams** (model, noise, loss, training, data, paths) | - Model: cover_size=256, message_length=100, max=256, embed_dim=64<br>- Training: epochs=60, batch=8, grad_accum=4, lr=1e-4, warmup=5, AMP=True, channels_last=True<br>- Validation: metric=BER (min), interval=1, save_images=True<br>- Data: dataset=custom/celeba, message_source=random_bits/text_file |

---

### 👤 Người 4 — Phần VI: Đánh giá & Phần VIII: Rủi ro
**Đọc 4 file này:**

| File | Chức năng | Điểm cần nắm cho báo cáo |
|------|-----------|--------------------------|
| `utils/metrics.py` | **Tất cả metrics**: PSNR, SSIM, LPIPS, BER, Bit Accuracy, Message Accuracy, Secret Reconstruction (PSNR/SSIM cho nhánh image) | - PSNR/SSIM: cover vs stego (chất lượng thị giác)<br>- BER: Bit Error Rate (%) = 100 - Bit Acc<br>- Bit Acc: % bit đúng sau sigmoid>0.5<br>- Message Acc: % sample recover đúng 100% bit<br>- Secret Reconstruction: PSNR/SSIM secret vs revealed (nhánh image) |
| `attacks/steganalysis.py` | **Steganalysis Attack**: SRNet/XuNet detector, fine-tune 5-10 epochs, AUC-ROC evaluation | - Load pretrained `weights/srnet.pth` (15MB), `weights/xunet.pth` (73KB)<br>- Fine-tune trên cover/stego của đề tài<br- Đánh giá AUC-ROC (mục tiêu ≈ 0.5 = random guess)<br>- Train/Val split cho detector |
| `attacks/robustness.py` | **Robustness Test**: JPEG(50-95), Gaussian(0.01-0.05), Crop(0.7-0.9), Rotation(±5-15°), Resize(0.5-2.0), Brightness/Contrast | - Test message recovery sau các biến đổi<br>- Trả về BER/Bit Acc cho từng noise level<br>- Dùng cho bảng robustness trong báo cáo |
| `evaluate.py` | **Pipeline đánh giá đầy đủ**: Load checkpoint → compute metrics → steganalysis → robustness → save figures + JSON | - `--steganalysis`: chạy SRNet/XuNet AUC<br>- `--robustness`: chạy robustness test<br>- `--save-figures`: lưu cover/stego/diff grid<br>- Output: `evaluation_results.json` + hình ảnh trong `outputs/` |

---

### 👤 Người 1 & 2 — Phần I, II, III (Không cần đọc code, chỉ dùng dàn ý)
**Tham khảo nhanh nếu muốn hiểu flow:**

| File | Tóm tắt |
|------|---------|
| `main.py` | CLI entry: `train-text`, `train-image`, `evaluate`, `demo` — parse args → gọi train.py/evaluate.py/gui |
| `train.py` | Vòng lặp train: forward → loss → backward (G + D riêng) → optimizer step → scheduler → validate → checkpoint |
| `data/dataset.py` | `TextStegoDataset`: cover image + random bits/text corpus → message bits; `CelebAStegoDataset`: auto-download CelebA |
| `models/discriminator.py` | PatchGAN 3-4 lớp SpectralNorm Conv, patch_size=70, LSGAN loss |
| `models/full_model.py` | Wrapper `TextStegoModel` (encoder + decoder_text + discriminator + noise) & `ImageStegoModel` |
| `gui/gradio_app.py` | Web demo: upload cover + nhập text → encode → hiển thị stego → decode → hiển thị text recover |

---

## 🔑 THAM SỐ CHỐT (Lấy từ `configs/text_stego.yaml`)

### Model Architecture
```yaml
cover_size: 256
message_length: 100          # Mặc định 100 bit (~12-15 ký tự ASCII)
max_message_length: 256
embed_dim: 64                # Per-bit embedding dimension
base_channels: 64            # U-Net base width
encoder_depth: 5             # 5 cấp downsample
use_residual_blocks: true
decoder_type: "conv_fc"      # 1x1 Conv + Flatten + FC
```

### Curriculum Noise (60 epochs)
```yaml
Phase 1 (0-15):   [Identity]
Phase 2 (15-30):  [Identity, JPEG(q=90), Gaussian(std=0.01)]
Phase 3 (30-45):  [+ Dropout(p=0.1), Crop(scale=0.9)]
Phase 4 (45-60):  [+ Resize(0.5-2.0), Rotation(10°), BrightnessContrast(0.2), JPEG(q=75), Gaussian(0.02), Dropout(0.2), Crop(0.85)]
```

### Loss Weights (Text Branch)
```yaml
lambda_pixel: 1.0       # MSE cover↔stego
lambda_lpips: 5.0       # LPIPS-AlexNet perceptual
lambda_adv: 0.5         # LSGAN adversarial
lambda_msg: 10.0        # BCE message reconstruction
```

### Training (VRAM 8GB optimized)
```yaml
epochs: 60
batch_size: 8
gradient_accumulation: 4      # Effective batch = 32
lr: 0.0001
lr_discriminator: 0.0001
scheduler: "cosine_warmup"
warmup_epochs: 5
mixed_precision: true         # FP16 AMP
channels_last: true           # NHWC memory format
compile_model: true           # torch.compile (cần MSVC trên Windows)
grad_clip: 1.0
```

### Validation
```yaml
interval: 1
metric: "ber"                 # Bit Error Rate (lower is better)
best_mode: "min"
save_images: true
```

---

## 📊 KẾT QUẢ MONG ĐỢI (Sau khi train xong 60 epochs trên CelebA)

| Metric | Target (Text Branch) | Ghi chú |
|--------|---------------------|---------|
| PSNR (cover vs stego) | > 30 dB | Chất lượng thị giác tốt |
| SSIM | > 0.95 | Cấu trúc giữ nguyên |
| LPIPS | < 0.1 | Perceptual similarity cao |
| BER | < 5% | Khôi phục bit chính xác |
| Bit Accuracy | > 95% | |
| Message Accuracy | > 90% | Toàn bộ chuỗi recover đúng |
| Steganalysis AUC (SRNet) | ≈ 0.50-0.55 | Gần random guess = an toàn |
| Robustness (JPEG75) | BER < 10% | Sau nén JPEG quality 75 |

---

## ⚠️ LƯU Ý KHI VIẾT BÁO CÁO

1. **Không viết số liệu suy diễn** — chờ tôi chạy xong train 60 epochs → gửi file `evaluation_results.json` + log TensorBoard
2. **Phần IV (Người 3)**: Vẽ sơ đồ kiến trúc dựa trên `encoder.py` + `decoder_text.py` + `noise_layers.py` (Encoder U-Net 5 tầng, Broadcaster, Noise layers pipeline, Curriculum 4 phase)
3. **Phần VI+VIII (Người 4)**: Bảng rủi ro map 1-1 metric:
   - OOM → batch_size/grad_accum (xem `train.py` line 350)
   - BER cao → tăng λ_msg, kéo dài Phase 1-2 (xem `configs/text_stego.yaml` noise curriculum)
   - AUC Steganalysis cao → tăng λ_adv, noise mạnh sớm hơn, Multi-scale Discriminator
   - Decoder Text GlobalAvgPool → đã fix bằng 1×1 Conv+FC (xem `decoder_text.py`)
4. **Terminology nhất quán**: "Cover image", "Stego image", "Message bits", "Bit Error Rate (BER)", "Curriculum Noise", "PatchGAN Discriminator", "LPIPS-AlexNet", "Straight-Through Estimator (STE)"

---

## 📞 LIÊN HỆ

- **Code owner / Trainer**: [Tên bạn] — chạy train, gửi kết quả, giải thích code
- **Deadline nội dung**: Theo timeline 3-4 tuần (xem dàn ý Word)

---

*File này tự động tạo từ source code review. Cập nhật khi code thay đổi.*