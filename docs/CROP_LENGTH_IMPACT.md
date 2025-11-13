# Crop Length 變化影響分析

## 從 256 改為 512 的詳細影響

### 📐 尺寸對比

| 項目 | crop_length=256 | crop_length=512 | 變化 |
|-----|----------------|----------------|-----|
| Piano roll 尺寸 | 128×256 | 128×512 | 寬度 ×2 |
| 總 patches 數 | 2,048 | 4,096 | ×2 |
| 時間範圍 | ~16 小節 | ~32 小節 | ×2 |

### 🧮 每層處理的序列長度

#### Level 0 (最粗層，128→128)
```
256: 128×256 → 32×64 = 2,048 patches
512: 128×512 → 32×128 = 4,096 patches  ⬆️ ×2
```

#### Level 1 (16→16)
```
256: 16×32 → 4×8 = 128 patches  
512: 16×64 → 4×16 = 256 patches  ⬆️ ×2
```

#### Level 2 (4→4)
```
256: 4×8 → 1×2 = 8 patches
512: 4×16 → 1×4 = 16 patches  ⬆️ ×2
```

#### Level 3 (1→1)
```
256: 1×2 → 1 patch
512: 1×4 → 1 patch
```

### 💾 記憶體需求變化

由於序列長度翻倍，記憶體需求會顯著增加：

| 配置 | Batch Size | Grad Accum | 估計 GPU 記憶體 |
|-----|-----------|------------|----------------|
| **256 (舊)** | 16 | 8 | ~12-16 GB |
| **512 (新)** | 16 | 16 | ~18-24 GB |

**為何需要更多記憶體？**
- Transformer attention 複雜度：O(n²)，其中 n 是序列長度
- 4096 patches 的 attention matrix 比 2048 patches 大 4 倍

### ⚙️ 配置調整建議

#### 更新的訓練配置 (`train_default.yaml`)

```yaml
training:
  accumulate_grad_batches: 16  # 從 8 增加到 16
  train_batch_size: 16         # 保持或減小
  val_batch_size: 4            # 減小以節省記憶體

model:
  grad_checkpointing: true     # 必須啟用！
```

#### 如果仍然 OOM，進一步調整：

```yaml
training:
  accumulate_grad_batches: 32  # 進一步增加
  train_batch_size: 8          # 減小 batch size
```

### 🎵 音樂內容變化

#### 時間範圍擴展

假設使用 16th note 作為時間單位：

```
256 steps = 16 小節 (4/4 拍)
512 steps = 32 小節 (4/4 拍)
```

**優點**：
- ✅ 可以捕捉更長的音樂結構
- ✅ 更完整的樂句和段落
- ✅ 更好的長期依賴建模

**挑戰**：
- ⚠️ 需要更多訓練數據
- ⚠️ 訓練時間增加
- ⚠️ 更難收斂

### 🔍 實際影響示例

#### Level 0 Generator (最重要的層)

**256 配置：**
```python
seq_len = 2048  # 需要處理的 patches
max_seq_len = 2048  # Transformer 支持的最大長度
```

**512 配置：**
```python
seq_len = 4096  # 需要處理的 patches  ⬆️
max_seq_len = 4096  # Transformer 支持的最大長度  ⬆️
```

這意味著：
- Self-attention 的計算量：2048² → 4096² (4倍)
- 位置編碼需要支持更長序列
- KV cache 在 inference 時需要更多記憶體

### 📊 訓練速度估算

| 項目 | crop_length=256 | crop_length=512 | 比例 |
|-----|----------------|----------------|-----|
| 單個 batch 時間 | 1.0x | ~2.5x | ⬆️ |
| Epoch 時間 | 1.0x | ~2.5x | ⬆️ |
| 收斂所需步數 | 基準 | 可能需要更多 | ? |

**為什麼不是 2x？**
- Attention 是 O(n²)，所以計算量增加超過線性
- 記憶體帶寬限制
- Gradient accumulation 增加

### 🎯 使用建議

#### 何時使用 512？

✅ **適合情況：**
- 有充足的 GPU 記憶體 (24GB+)
- 數據集有長序列音樂
- 需要建模長期結構（如主歌-副歌）
- 最終生成目標是完整樂段

#### 何時使用 256？

✅ **適合情況：**
- GPU 記憶體有限 (16GB)
- 快速實驗和調試
- 數據集主要是短片段
- 重點在和弦進行而非整體結構

#### 何時使用 128？

✅ **適合情況：**
- 極度受限的硬體
- 快速原型開發
- 學習和測試
- 小模型訓練

### 🔧 程式碼中的對應變化

#### `models/fractal_gen.py`

```python
# 舊設定 (最大支持 256)
max_w_patches = 256 // img_size_list[fractal_level+1]

# 新設定 (最大支持 512)
max_w_patches = 512 // img_size_list[fractal_level+1]  ✓
```

這確保 Transformer 的位置編碼和 attention mask 能支持更長序列。

#### `inference.py`

```python
# 預設生成長度
default=256  →  default=512  ✓
```

確保 inference 時生成的長度與訓練時一致。

### ⚠️ 潛在問題

#### 1. Position Embedding 限制

如果模型的 position embedding 固定長度不足：
```python
# 需要確保支持 4096 positions
max_positions = 4096  # 或更大
```

#### 2. Attention Mask

確保 attention mask 計算支持新的序列長度。

#### 3. 數據處理

```python
# dataset.py 中的處理
if piano_roll.shape[1] < crop_length:
    # 需要處理短於 512 的數據
    pad_length = crop_length - piano_roll.shape[1]
    # ... padding logic
```

### 📈 預期效果

#### 訓練收斂

```
crop_length=256: 收斂步數 ~50K steps
crop_length=512: 收斂步數 ~80K-100K steps  (估計)
```

#### 生成質量

- **短期結構**：差異不大
- **長期結構**：512 應該明顯更好
- **音樂連貫性**：512 能捕捉更長的主題重複

### 🚀 遷移步驟

如果從 256 遷移到 512：

1. **更新配置文件**：`crop_length: 512`
2. **調整記憶體設置**：增加 `accumulate_grad_batches`
3. **啟用 checkpointing**：`grad_checkpointing: true`
4. **測試記憶體**：先用 `fast_dev_run: true` 測試
5. **開始訓練**：監控記憶體使用和訓練速度
6. **調整 inference**：使用 `generation_length: 512`

### 📝 檢查清單

在切換到 512 之前：

- [ ] GPU 記憶體 ≥ 20GB
- [ ] `accumulate_grad_batches` ≥ 16
- [ ] `grad_checkpointing: true`
- [ ] 數據集有足夠長的序列
- [ ] 更新了 `max_w_patches` 至 512
- [ ] 測試過 OOM 不會發生
- [ ] Inference 配置也更新到 512

### 🎓 總結

**Crop length 512 vs 256 的核心差異：**

| 維度 | 影響 | 幅度 |
|-----|-----|-----|
| 輸入尺寸 | 增加 | ×2 |
| Patches 數 | 增加 | ×2 |
| 計算量 | 增加 | ~×2.5 |
| 記憶體需求 | 增加 | ~×1.5-2 |
| 訓練時間 | 增加 | ~×2.5 |
| 音樂長度 | 增加 | ×2 |
| 長期結構 | 改善 | 顯著 |

**結論**：512 提供更好的音樂長期結構建模能力，但需要更多計算資源和訓練時間。

