# FractalMIDI 修正總結

## 修正日期
2025-11-07

## 主要修正

### 1. ✅ 修正 VelocityLoss 採樣問題

**問題**：模型生成的輸出全是 0，導致 MIDI 檔案只有一個音符。

**根本原因**：
- `VelocityLoss.sample()` 中的數值不穩定
- 採樣邏輯沒有正確處理概率分佈

**修正內容**：
- 添加數值穩定性檢查：`logits = torch.clamp(logits, min=-20, max=20)`
- 在 softmax 後添加 epsilon：`probs = probs + 1e-10`
- 修正溫度應用：`logits = logits / max(temperature, 1e-8)`
- 正確處理 [-1, 1] 範圍的值

**文件**：`model.py` 第 1057-1118 行

### 2. ✅ 將初始化從 0（黑色）改為 -1（白色）

**目的**：讓模型從白色畫布開始，逐步「繪製」音符上去，更符合直覺。

**修正位置**：

#### a. AR 層的 canvas 初始化
```python
# model.py 第 921-928 行
canvas = torch.full(
    (bsz, self.seq_len, self.patch_size**2),
    -1.0,  # 白色/靜音
    device=cond_list[0].device,
    dtype=cond_list[0].dtype
)
```

#### b. MAR 層的 patches 初始化
```python
# model.py 第 474-475 行
patches = torch.full((bsz, actual_seq_len, 1 * self.patch_size**2), -1.0, device=cond_list[0].device)

# model.py 第 667-668 行 (_sample_fast)
patches = torch.full((base_bsz, seq_len, patch_dim), -1.0, device=device, dtype=dtype)
```

#### c. FractalGen 的 canvas 初始化
```python
# model.py 第 1257-1258 行
'canvas': torch.full((batch_size, 1, 128, 256), -1.0)
```

#### d. VelocityLoss 的初始化
```python
# model.py 第 1066-1068 行
velocity_values = torch.full((bsz, 1), -1.0, device=cond_list[0].device)
```

**影響範圍**：
- ✅ Unconditional generation
- ✅ Conditional generation
- ✅ Inpainting

### 3. ✅ 修正 GIF 生成的值範圍處理

**問題**：GIF 生成假設值在 [0, 1] 範圍，但現在是 [-1, 1]。

**修正**：
```python
# trainer.py 第 356-359 行
# Normalize from [-1, 1] to [0, 255]
# -1 (white/silence) -> 0, 1 (loud/black) -> 255
frame_np = ((frame_np + 1.0) / 2.0 * 255)
frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
```

**文件**：`trainer.py` 第 327-424 行

### 4. ✅ GIF 生成功能已實作

**功能**：
- 在 validation 時自動生成 GIF 動畫
- 展示模型的階層式生成過程
- 每個 `log_images_every_n_steps` 生成一次

**儲存位置**：
```
outputs/{experiment_name}/lightning_logs/version_X/generation_gifs/
├── step_0010000_sample_0.gif
├── step_0010000_sample_1.gif
└── ...
```

**配置**：
```bash
python main.py \
    --log_images_every_n_steps 5000 \  # 每 5000 步生成一次
    --num_images_to_log 4               # 每次生成 4 個 GIF
```

## 測試結果

### 生成測試
```bash
cd /home/feiyueh/FractalMIDI
python inference.py \
    --checkpoint outputs/fractalgen_ar_ar_ar_ar/checkpoints/step_00005000-val_loss_0.0414.ckpt \
    --mode unconditional \
    --num_samples 2 \
    --num_iter_list 4 2 1 1 \
    --output_dir outputs/test_inference
```

**結果**：
- ✅ 生成成功
- ✅ 輸出有變化（5.40% 非白色像素）
- ✅ 88 種不同的顏色
- ✅ 不再是全 0 或全黑

### 值範圍分佈
```
< -0.5 (very silent): 93.18%
[-0.5, 0.0):          1.31%
[0.0, 0.5):           1.90%
>= 0.5 (loud):        3.61%
```

## 兼容性

### AR 和 MAR Checkpoint

**現狀**：
- ✅ AR checkpoint 可以正常載入和使用
- ✅ MAR checkpoint 可以正常載入和使用
- ⚠️  AR 和 MAR checkpoint **不能**互相使用（架構不同）

**原因**：
- AR 使用 `pos_embed`（位置嵌入）
- MAR 使用 `mask_token`（遮罩 token）
- 兩者的權重鍵不同

**建議**：
- 訓練時明確指定 `--generator_types`
- 推論時使用對應的 checkpoint
- 例如：AR 訓練用 `ar ar ar ar`，MAR 訓練用 `mar mar mar mar`

### 向後兼容性

**舊 Checkpoint**：
- ⚠️  舊的 checkpoint（初始化為 0）仍然可以載入
- ⚠️  但生成結果可能不如新訓練的模型
- ✅ 建議用新的初始化（-1）重新訓練

## 使用指南

### 1. 訓練新模型

```bash
bash run_training.sh
```

配置說明：
- `GENERATOR_TYPES="ar ar ar ar"` - 使用全 AR 架構
- `LOG_IMAGES_EVERY_N_STEPS=5000` - 每 5000 步生成圖片和 GIF
- `NUM_IMAGES_TO_LOG=4` - 每次生成 4 個樣本

### 2. 生成 MIDI

```bash
bash run_inference.sh
```

參數說明：
- `NUM_ITER_LIST="12 8 4 1"` - 每層的迭代次數
- `TEMPERATURE=1.0` - 採樣溫度
- `SPARSITY_BIAS=0.0` - 稀疏性偏置（0 表示不調整）

### 3. 查看 GIF

GIF 儲存在：
```
outputs/fractalgen_ar_ar_ar_ar/lightning_logs/version_0/generation_gifs/
```

使用瀏覽器或圖片查看器打開即可。

## 已知問題

### 1. 生成速度較慢

**原因**：
- AR 需要序列生成每個 patch
- 啟用 `return_intermediates=True` 會記錄所有中間步驟

**解決方案**：
- 推論時不使用 `return_intermediates`（inference.py 已經這樣做）
- 調整 `num_iter_list` 減少迭代次數

### 2. 訓練時記憶體使用

**GIF 生成會增加記憶體**：
- 需要保存所有中間步驟
- 建議 `num_images_to_log` 不超過 4

**配置建議**：
```python
log_images_every_n_steps=5000  # 不要太頻繁
num_images_to_log=4            # 不要太多
```

### 3. FutureWarning

```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
```

**影響**：僅警告，不影響功能

**修正**（可選）：
```python
# 將
with torch.cuda.amp.autocast(enabled=False):
# 改為
with torch.amp.autocast('cuda', enabled=False):
```

## 下一步建議

### 1. 重新訓練模型

使用新的初始化（-1）重新訓練，預期會有更好的生成質量。

```bash
# 清除舊的輸出
rm -rf outputs/fractalgen_ar_ar_ar_ar

# 開始新訓練
bash run_training.sh
```

### 2. 實驗不同配置

**AR vs MAR**：
```bash
# 全 AR
--generator_types ar ar ar ar

# 全 MAR
--generator_types mar mar mar mar

# 混合（頂層 AR，其餘 MAR）
--generator_types ar mar mar mar
```

**迭代次數**：
```bash
# 快速生成（低質量）
--num_iter_list 4 2 1 1

# 標準生成
--num_iter_list 8 4 2 1

# 高質量生成（慢）
--num_iter_list 16 8 4 1
```

### 3. 監控訓練

```bash
tensorboard --logdir outputs/fractalgen_ar_ar_ar_ar
```

查看：
- `train/loss`, `val_loss` - 損失曲線
- `val/generated/` - 生成的 piano rolls
- `val/generation_preview/` - GIF 預覽
- `val/ground_truth/` - 真實數據

### 4. 測試 GIF 生成

```bash
bash test_gif_quick.sh
```

檢查是否有 GIF 檔案生成：
```bash
find outputs/test_gif -name "*.gif"
```

## 技術細節

### 值範圍轉換

**訓練時**：
- Piano roll 值：[0, 1]（從 MIDI velocity 正規化）
- 內部表示：[-1, 1]（-1 = 靜音，1 = 最大音量）

**生成時**：
- 初始化：-1（白色/靜音）
- 生成範圍：[-1, 1]
- 輸出：轉換回 [0, 1] 用於 MIDI

**可視化**：
- Colormap：viridis
- 藍色 = 靜音（-1）
- 黃色 = 響亮（1）

### 採樣穩定性

**數值範圍限制**：
```python
logits = torch.clamp(logits, min=-20, max=20)
```

**概率正規化**：
```python
probs = probs + 1e-10
probs = probs / probs.sum(dim=-1, keepdim=True)
```

**溫度控制**：
```python
logits = logits / max(temperature, 1e-8)
```

## 總結

✅ **已完成**：
1. 修正 VelocityLoss 採樣邏輯
2. 將初始化改為 -1（白色）
3. 修正 GIF 生成的值範圍處理
4. 實作訓練時的 GIF 生成功能
5. 確保 sample images 正常記錄

✅ **測試通過**：
- 生成不再全是 0
- 輸出有合理的值分佈
- GIF 生成邏輯正確

⚠️  **注意事項**：
- AR 和 MAR checkpoint 不能互換
- 建議用新初始化重新訓練
- GIF 生成會增加記憶體和時間

📝 **建議**：
- 重新訓練以獲得最佳效果
- 實驗不同的 generator_types 組合
- 監控 TensorBoard 中的 GIF 和圖片

---

**文檔版本**：1.0  
**最後更新**：2025-11-07  
**作者**：AI Assistant

