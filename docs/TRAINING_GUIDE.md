# FractalMIDI 訓練指南

## 快速開始

### 1. 測試訓練設置

首先運行快速測試確認環境正常：

```bash
bash train_test.sh
```

這會運行幾個訓練步驟來驗證：
- 資料載入正常
- 模型可以正常前向傳播
- GPU 記憶體足夠
- 所有依賴正確安裝

### 2. 選擇模型大小

根據您的需求選擇模型大小：

#### 128×128 模型（推薦入門）

**特點：**
- 訓練尺寸：128 pitches × 128 time steps
- 時長：8 measures (4/4 time)
- 記憶體需求：~4-6GB per GPU
- 訓練速度：較快
- 適用場景：短旋律、動機、快速迭代

**訓練命令：**
```bash
bash train_128x128.sh
```

**或手動指定：**
```bash
python main.py \
    --crop_length 128 \
    --train_batch_size 16 \
    --output_dir outputs/my_128x128_model
```

#### 256×256 模型（實際 128×512）

**特點：**
- 訓練尺寸：128 pitches × 512 time steps
- 顯示格式：256×256（視覺化用途，垂直堆疊兩段）
- 時長：32 measures (4/4 time)
- 記憶體需求：~12-16GB per GPU
- 訓練速度：較慢
- 適用場景：完整樂句、長期音樂結構

**訓練命令：**
```bash
bash train_256x256.sh
```

**或手動指定：**
```bash
python main.py \
    --crop_length 512 \
    --train_batch_size 4 \
    --output_dir outputs/my_256x256_model
```

## 完整命令列參數

### 資料相關

```bash
--train_data dataset/train.txt        # 訓練資料列表
--val_data dataset/valid.txt          # 驗證資料列表
--crop_length 256                     # 裁剪長度（128/256/512）
--augment_factor 1                    # 每個 MIDI 生成幾個隨機裁剪
--pitch_shift_min -3                  # 音高偏移最小值（半音）
--pitch_shift_max 3                   # 音高偏移最大值（半音）
```

### 訓練超參數

```bash
--train_batch_size 8                  # 訓練 batch size
--val_batch_size 8                    # 驗證 batch size
--max_steps 200000                    # 最大訓練步數
--lr 1e-4                             # 學習率
--warmup_steps 2000                   # Warmup 步數
--weight_decay 0.05                   # Weight decay
--grad_clip 3.0                       # Gradient clipping
--accumulate_grad_batches 1           # Gradient accumulation
```

### 模型配置

```bash
--generator_types "mar,mar,mar,mar"   # 每層的生成器類型（mar 或 ar）
--scan_order "row_major"              # AR 掃描順序（row_major 或 column_major）
--mask_ratio_loc 1.0                  # MAR mask ratio 平均值
--mask_ratio_scale 0.5                # MAR mask ratio 標準差
```

### 硬體設置

```bash
--devices "0,1"                       # GPU 索引（逗號分隔）
--num_workers 4                       # DataLoader workers
--precision "32"                      # 訓練精度（32/16/bf16）
--grad_checkpoint                     # 啟用 gradient checkpointing（省記憶體）
```

### Logging 和 Checkpointing

```bash
--output_dir outputs/my_model         # 輸出目錄
--log_every_n_steps 50                # Logging 頻率
--val_check_interval_steps 2000       # 驗證頻率
--checkpoint_every_n_steps 5000       # Checkpoint 頻率
--log_images_every_n_steps 5000       # 生成樣本頻率
--save_top_k 3                        # 保留最好的 k 個 checkpoints
```

## Generator 類型配置

每個層級可以獨立選擇生成器類型：

### MAR (Masked Autoregressive)
- **優點**：並行生成、訓練穩定、品質高
- **缺點**：需要多次迭代
- **適用**：大部分場景

### AR (Autoregressive)
- **優點**：序列化生成、細節控制好
- **缺點**：慢、容易累積錯誤
- **適用**：需要精確控制生成順序時

### 建議配置

```bash
# 全 MAR（推薦）
--generator_types "mar,mar,mar,mar"

# 全 AR
--generator_types "ar,ar,ar,ar"

# 混合（高層 MAR，低層 AR）
--generator_types "mar,mar,ar,ar"
```

## Scan Order（僅 AR）

對於 AR 生成器，可以選擇掃描順序：

### row_major（行優先）
- 掃描順序：先掃描完一行，再換下一行
- 對於 piano roll (128, W)：先掃描完所有時間步，再換音高
- **適合**：旋律線、單音序列

### column_major（列優先）
- 掃描順序：先掃描完一列，再換下一列
- 對於 piano roll (128, W)：先掃描完所有音高，再換時間步
- **適合**：和弦結構、垂直和聲

```bash
# 建議嘗試兩種都訓練比較
bash train_128x128.sh  # 預設 row_major
# 修改腳本中的 SCAN_ORDER="column_major" 再訓練一次比較
```

## 記憶體優化

如果遇到 OOM (Out of Memory)：

### 1. 減少 batch size
```bash
--train_batch_size 4  # 或更小
```

### 2. 啟用 gradient checkpointing
```bash
--grad_checkpoint
```
注意：會稍微降低訓練速度

### 3. 使用混合精度
```bash
--precision "bf16"  # 或 "16"
```

### 4. 減少 DataLoader workers
```bash
--num_workers 2
```

### 5. 禁用 in-memory cache
```bash
--no_cache_in_memory
```

## 訓練監控

### TensorBoard

```bash
tensorboard --logdir outputs/my_model/logs
```

查看：
- Loss curves
- Learning rate schedule
- Generated samples
- 每層的統計資訊

### 檢查 Checkpoints

Checkpoints 儲存在：
```
outputs/my_model/checkpoints/
├── step_00005000-val_loss_0.0234.ckpt
├── step_00010000-val_loss_0.0198.ckpt
└── ...
```

## 從 Checkpoint 恢復訓練

如果訓練中斷，可以從 checkpoint 恢復：

```bash
python main.py \
    --crop_length 256 \
    --train_batch_size 8 \
    --output_dir outputs/my_model \
    --resume_from_checkpoint outputs/my_model/checkpoints/latest.ckpt
```

## 訓練後測試

訓練完成後，使用推理腳本測試：

```bash
# 對於 128×128 模型
python inference.py \
    --checkpoint outputs/my_128x128_model/checkpoints/best.ckpt \
    --target_width 128 \
    --num_samples 10 \
    --output_dir outputs/test_generation

# 對於 256×256 模型（128×512）
python inference.py \
    --checkpoint outputs/my_256x256_model/checkpoints/best.ckpt \
    --target_width 512 \
    --num_samples 10 \
    --output_dir outputs/test_generation
```

## 常見問題

### Q: 應該選擇哪個模型大小？

**A:** 
- 如果是第一次訓練：選 128×128，訓練快、除錯容易
- 如果需要長序列：選 256×256（512）
- 可以先訓練 128×128，再 fine-tune 到更大尺寸

### Q: 訓練需要多久？

**A:**
- 128×128：2-4 天（雙 GPU，200k steps）
- 256×256：4-7 天（雙 GPU，200k steps）

### Q: 如何知道訓練是否收斂？

**A:** 觀察：
- Validation loss 穩定下降
- Generated samples 品質逐漸提升
- 各層的 loss contribution 趨於穩定

### Q: 可以改變 crop_length 嗎？

**A:** 可以！支援任何是 4 的倍數的長度：
```bash
--crop_length 64   # 最小
--crop_length 128  # 標準小
--crop_length 256  # 標準中
--crop_length 512  # 標準大
--crop_length 1024 # 超大（需要大量記憶體）
```

### Q: MAR 和 AR 哪個好？

**A:** 
- **MAR**：通常品質更好、訓練更穩定（推薦）
- **AR**：適合需要精確控制生成順序的場景
- 建議先全用 MAR，有需要再嘗試混合

## 範例工作流程

### 完整訓練流程

```bash
# 1. 測試環境
bash train_test.sh

# 2. 訓練 128×128 模型（快速原型）
bash train_128x128.sh

# 3. 測試生成
python inference.py \
    --checkpoint outputs/fractalgen_128x128/checkpoints/step_00100000-val_loss_0.0123.ckpt \
    --target_width 128 \
    --num_samples 20

# 4. 如果效果好，訓練更大模型
bash train_256x256.sh

# 5. 最終測試
python inference.py \
    --checkpoint outputs/fractalgen_256x256/checkpoints/step_00150000-val_loss_0.0089.ckpt \
    --target_width 512 \
    --num_samples 50 \
    --temperature 0.9
```

## 進階技巧

### 1. 兩階段訓練

先訓練小模型，再 fine-tune 到大尺寸：

```bash
# Stage 1: Train on 128×128
python main.py --crop_length 128 --max_steps 100000 --output_dir outputs/stage1

# Stage 2: Fine-tune on 256×256
python main.py --crop_length 256 --max_steps 50000 \
    --resume_from_checkpoint outputs/stage1/checkpoints/latest.ckpt \
    --output_dir outputs/stage2
```

### 2. 實驗不同配置

```bash
# Experiment 1: All MAR
python main.py --crop_length 256 --generator_types "mar,mar,mar,mar" \
    --output_dir outputs/exp_mar

# Experiment 2: All AR with row_major
python main.py --crop_length 256 --generator_types "ar,ar,ar,ar" \
    --scan_order "row_major" --output_dir outputs/exp_ar_row

# Experiment 3: All AR with column_major
python main.py --crop_length 256 --generator_types "ar,ar,ar,ar" \
    --scan_order "column_major" --output_dir outputs/exp_ar_col
```

### 3. 資料增強調整

```bash
# Heavy augmentation
python main.py --crop_length 256 \
    --augment_factor 4 \
    --pitch_shift_min -6 \
    --pitch_shift_max 6

# Minimal augmentation
python main.py --crop_length 256 \
    --augment_factor 1 \
    --pitch_shift_min 0 \
    --pitch_shift_max 0
```

## 檔案結構

訓練後的輸出結構：

```
outputs/my_model/
├── checkpoints/
│   ├── step_00005000-val_loss_0.0234.ckpt
│   ├── step_00010000-val_loss_0.0198.ckpt
│   └── ...
└── logs/
    └── version_0/
        ├── events.out.tfevents...  # TensorBoard logs
        └── hparams.yaml            # Hyperparameters
```

祝訓練順利！🎵

