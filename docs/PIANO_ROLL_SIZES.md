# Piano Roll 尺寸配置指南

## 概述

FractalMIDI 模型的 **pitch 維度固定為 128**，對應 MIDI 的完整音域（21-108）。
時間維度可以根據需求調整，常見的配置有：

| 配置文件 | 尺寸 | Patches | 用途 |
|---------|------|---------|------|
| `train_128x128.yaml` | 128×128 | 1024 (32×8) | 快速測試、小模型訓練 |
| `train_128x256.yaml` | 128×256 | 2048 (32×16) | 中等長度，平衡速度與質量 |
| `train_default.yaml` | 128×512 | 4096 (32×32) | **預設**，適合完整音樂片段 |

## 配置文件設置

### 訓練配置

所有配置文件的關鍵參數：

```yaml
# Model configuration
model:
  img_size_list: [128, 16, 4, 1]  # 固定不變，定義 4 層架構

# Data configuration
data:
  crop_length: 512  # 時間維度長度，可改為 128, 256, 512
```

### Inference 配置

對應的 inference 配置：

```yaml
# config/inference_default.yaml (128×512)
unconditional:
  length: 512  # 生成長度

# config/inference_small.yaml (128×128，用於小模型)
unconditional:
  length: 128  # 生成長度
```

## 模型序列長度限制

在 `models/fractal_gen.py` 中，`max_seq_len` 計算如下：

```python
h_patches = 128 // img_size_list[fractal_level+1]
max_w_patches = 512 // img_size_list[fractal_level+1]  # 支持最大 512 寬度
expected_seq_len = h_patches * max_w_patches
```

對於不同的配置：

### 128×512（預設）
- Level 0: 128×512 → 2048 patches → expected_seq_len = 4096
- Level 1: 16×64 → 256 patches → expected_seq_len = 512
- Level 2: 4×16 → 16 patches → expected_seq_len = 64
- Level 3: 1×4 → 1 patch → expected_seq_len = 4

### 128×256
- Level 0: 128×256 → 1024 patches → expected_seq_len = 2048
- Level 1: 16×32 → 128 patches → expected_seq_len = 256
- Level 2: 4×8 → 8 patches → expected_seq_len = 32
- Level 3: 1×2 → 1 patch → expected_seq_len = 2

### 128×128
- Level 0: 128×128 → 512 patches → expected_seq_len = 1024
- Level 1: 16×16 → 64 patches → expected_seq_len = 128
- Level 2: 4×4 → 4 patches → expected_seq_len = 16
- Level 3: 1×1 → 1 patch → expected_seq_len = 1

## 更改尺寸的步驟

### 1. 選擇或創建配置文件

```bash
# 使用預設 128×512
python main.py --config config/train_default.yaml

# 使用 128×256
python main.py --config config/train_128x256.yaml

# 使用 128×128
python main.py --config config/train_128x128.yaml
```

### 2. 自定義配置

創建新的配置文件，修改 `crop_length`：

```yaml
data:
  crop_length: 1024  # 例如：128×1024
```

**注意**：如果 `crop_length > 512`，需要修改 `models/fractal_gen.py` 中的 `max_w_patches`：

```python
# 例如支持 1024 寬度
max_w_patches = 1024 // img_size_list[fractal_level+1]
```

### 3. Inference 時匹配訓練尺寸

確保 inference 時使用的 `generation_length` 和 `target_width` 與訓練時的 `crop_length` 一致：

```bash
# 對於 128×512 訓練的模型
python inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --generation_length 512 \
    --target_width 512

# 對於 128×256 訓練的模型
python inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --generation_length 256 \
    --target_width 256
```

## 記憶體使用

不同尺寸的記憶體需求（近似值，取決於 batch size 和其他設置）：

| 尺寸 | Batch Size | Grad Accumulation | 記憶體使用 (每 GPU) |
|-----|------------|-------------------|---------------------|
| 128×128 | 32 | 1 | ~8 GB |
| 128×256 | 16 | 8 | ~16 GB |
| 128×512 | 16 | 16 | ~20 GB |

**建議**：
- 128×128：適合 GPU 記憶體較小的情況
- 128×256：平衡選項，適合大多數 GPU
- 128×512：需要較大 GPU 記憶體（建議 24GB+）

## 最佳實踐

1. **保持 pitch 維度為 128**：這是 MIDI 的標準音域，不應改變
2. **根據硬體選擇時間維度**：記憶體越大，可以選擇越長的序列
3. **訓練和 inference 尺寸一致**：避免維度不匹配的問題
4. **漸進式訓練**：可以先用 128×128 快速實驗，再用 128×512 完整訓練
5. **數據增強**：較小尺寸時增加 `augment_factor` 來提高數據多樣性

## 故障排除

### 問題：OOM (Out of Memory) 錯誤

**解決方案**：
1. 減小 `crop_length`（512 → 256 → 128）
2. 減小 `train_batch_size`
3. 增加 `accumulate_grad_batches`
4. 啟用 `grad_checkpointing: true`

### 問題：生成結果維度不匹配

**解決方案**：
確保 inference 時的 `target_width` 與訓練時的 `crop_length` 一致。

### 問題：序列太長，超出 max_seq_len

**解決方案**：
在 `models/fractal_gen.py` 中增加 `max_w_patches`。

## 更新日誌

- **2024-11**: 默認配置從 128×256 更新為 128×512
- 所有配置文件的 pitch 維度統一為 128
- 添加 `train_128x256.yaml` 以保留舊配置

