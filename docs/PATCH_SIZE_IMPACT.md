# Patch Size 影響分析

## 概述

`patch_size` 定義了將 piano roll 劃分為 tokens 的基本單位。這是一個核心架構參數，直接影響模型的計算效率、記憶體使用和生成質量。

**當前默認值**: `patch_size = 4`

---

## 📐 基本概念

### Patch 劃分方式

Piano roll 被劃分為 `patch_size × patch_size` 的方形 patches：

```
Piano Roll: 128 (pitch) × 512 (time)
           ↓ patch_size=4
Patches:   32 (pitch_patches) × 128 (time_patches) = 4096 patches
```

每個 patch 包含：
- **Pitch 維度**: `patch_size` 個音高
- **Time 維度**: `patch_size` 個時間步
- **總像素數**: `patch_size²` 個 velocity 值

---

## 🔢 不同 Patch Size 的比較

### 128×512 Piano Roll 示例

| patch_size | Patch 數量 | 序列長度 | Pitch 分辨率 | Time 分辨率 | 每 patch 像素數 |
|-----------|-----------|---------|-------------|------------|---------------|
| 2 | 64×256 = 16,384 | 16,384 | 2 semitones | 2 steps | 4 |
| **4** | **32×128 = 4,096** | **4,096** | **4 semitones** | **4 steps** | **16** |
| 8 | 16×64 = 1,024 | 1,024 | 8 semitones | 8 steps | 64 |
| 16 | 8×32 = 256 | 256 | 16 semitones | 16 steps | 256 |

### 視覺化

```
patch_size=2:
████████████████████ 16,384 patches (極細)
計算量: 非常大
分辨率: 極高

patch_size=4 (默認):
████████████ 4,096 patches (平衡)
計算量: 適中
分辨率: 高

patch_size=8:
██████ 1,024 patches (粗)
計算量: 小
分辨率: 中等

patch_size=16:
███ 256 patches (極粗)
計算量: 很小
分辨率: 低
```

---

## 📊 詳細影響分析

### 1. **序列長度變化**

序列長度 = (piano_roll_height / patch_size) × (piano_roll_width / patch_size)

```python
# 對於 128×512 piano roll
patch_size=2:  seq_len = (128/2) × (512/2) = 64 × 256 = 16,384
patch_size=4:  seq_len = (128/4) × (512/4) = 32 × 128 = 4,096  ✓ 當前
patch_size=8:  seq_len = (128/8) × (512/8) = 16 × 64 = 1,024
patch_size=16: seq_len = (128/16) × (512/16) = 8 × 32 = 256
```

**影響**:
- ✅ 序列長度 ↓ = 計算速度 ↑
- ⚠️ 序列長度 ↓ = 分辨率 ↓

### 2. **計算複雜度變化**

Transformer attention 複雜度: **O(n²)**

```python
# 相對計算量 (以 patch_size=4 為基準)
patch_size=2:  (16384/4096)² = 16x  🔴 非常慢
patch_size=4:  1x (基準)           ✓ 平衡
patch_size=8:  (1024/4096)² = 1/16x ✅ 快 16 倍
patch_size=16: (256/4096)² = 1/256x ✅ 快 256 倍
```

**對於單個 batch**:
- `patch_size=2`: ~16 倍計算時間
- `patch_size=4`: 基準
- `patch_size=8`: ~1/16 計算時間
- `patch_size=16`: ~1/256 計算時間

### 3. **記憶體使用變化**

Attention matrix 大小: **seq_len × seq_len**

```python
# 128×512 piano roll, attention matrix 大小
patch_size=2:  16,384 × 16,384 = 268M elements  🔴 ~1 GB
patch_size=4:  4,096 × 4,096 = 16.7M elements    ✓ ~67 MB
patch_size=8:  1,024 × 1,024 = 1.05M elements    ✅ ~4 MB
patch_size=16: 256 × 256 = 65K elements          ✅ ~0.26 MB
```

**GPU 記憶體需求** (估算，包含其他開銷):
- `patch_size=2`: 可能需要 40+ GB (不可行)
- `patch_size=4`: 約 20-24 GB ✓
- `patch_size=8`: 約 8-12 GB ✅
- `patch_size=16`: 約 4-6 GB ✅✅

### 4. **分辨率影響**

#### Pitch 分辨率

```
patch_size=2: 2 semitones/patch
  → 可以區分相鄰半音
  → 和弦細節豐富

patch_size=4: 4 semitones/patch  ✓
  → 大約 1/3 八度
  → 平衡的和弦表達

patch_size=8: 8 semitones/patch
  → 超過半個八度
  → 和弦可能模糊

patch_size=16: 16 semitones/patch
  → 超過一個八度
  → 和弦嚴重模糊
```

#### Time 分辨率

假設 16th note 為時間單位：

```
patch_size=2: 2 steps = 1/8 note
  → 可表達快速音符

patch_size=4: 4 steps = 1/4 note  ✓
  → 適合大部分節奏

patch_size=8: 8 steps = 1/2 note
  → 只能表達較慢音符

patch_size=16: 16 steps = 1 whole note
  → 只能表達很慢音符
```

### 5. **層次結構影響**

FractalMIDI 使用 4 層架構: `[128, 16, 4, 1]`

這意味著每層之間是 **8 倍下采樣** (128→16, 16→4, 4→1):

```python
# patch_size 必須整除 img_size_list
img_size_list = [128, 16, 4, 1]

patch_size=2: ✅ 可以 (128/2=64, 16/2=8, 4/2=2, 1/2=0.5 ✗)
              → 需要調整架構為 [128, 16, 8, 4, 2]
              
patch_size=4: ✅ 可以 (128/4=32, 16/4=4, 4/4=1, 1/4=0.25 ✗)
              → 當前架構完美適配
              
patch_size=8: ✅ 可以 (128/8=16, 16/8=2, 4/8=0.5 ✗, ...)
              → 需要調整架構為 [128, 16, 8]
              
patch_size=16: ✅ 可以 (128/16=8, 16/16=1, 4/16=0.25 ✗, ...)
               → 需要調整架構為 [128, 16]
```

⚠️ **重要**: 改變 patch_size 通常需要同時調整 `img_size_list`！

---

## 🎵 音樂質量影響

### Patch Size = 2 (極細)

**優點**:
- ✅ 極高的音高精度
- ✅ 極高的時間精度
- ✅ 可以捕捉非常細微的音符變化
- ✅ 複雜的和弦和旋律細節

**缺點**:
- ❌ 計算量極大 (可能不可訓練)
- ❌ 記憶體需求極高
- ❌ 訓練時間極長
- ❌ 可能過擬合小細節

**適合**:
- 學術研究
- 極度關注細節的場景
- 有超大 GPU 資源

### Patch Size = 4 (當前默認) ⭐

**優點**:
- ✅ 平衡的音高精度 (4 semitones)
- ✅ 平衡的時間精度 (1/4 note)
- ✅ 適中的計算量
- ✅ 可以表達大部分音樂元素
- ✅ 與當前架構完美匹配

**缺點**:
- ⚠️ 可能無法捕捉極快速的音符
- ⚠️ 緊密的半音和弦可能略模糊

**適合**:
- ✅ 大部分音樂生成任務
- ✅ 平衡質量和效率
- ✅ 標準 GPU (24GB)

### Patch Size = 8 (粗)

**優點**:
- ✅ 計算速度快 (16倍)
- ✅ 記憶體需求低
- ✅ 可以處理更長序列
- ✅ 快速原型開發

**缺點**:
- ❌ 音高精度降低 (8 semitones, 超過半個八度)
- ❌ 時間精度降低 (1/2 note)
- ❌ 和弦細節丟失
- ❌ 快速音符無法表達

**適合**:
- 快速實驗
- 硬體資源有限
- 只關注整體結構的場景
- 慢速音樂 (ambient, drone)

### Patch Size = 16 (極粗)

**優點**:
- ✅ 極快的計算速度
- ✅ 極低的記憶體需求
- ✅ 可以處理極長序列

**缺點**:
- ❌ 音高精度極低 (16 semitones, 超過一個八度)
- ❌ 時間精度極低 (whole note)
- ❌ 幾乎無法表達音樂細節
- ❌ 生成質量嚴重降低

**適合**:
- 僅用於架構測試
- 極度資源受限
- 不推薦用於實際音樂生成

---

## 🔧 如何更改 Patch Size

### 步驟 1: 更新配置文件

```yaml
# config/train_custom.yaml
model:
  patch_size: 8  # 改為 8
  
  # 需要調整 img_size_list 以匹配
  img_size_list: [128, 16, 8]  # 移除 [4, 1]
  embed_dim_list: [512, 256, 128]  # 相應減少層數
  num_blocks_list: [12, 3, 2]
  num_heads_list: [8, 4, 2]
```

### 步驟 2: 驗證兼容性

```python
# 檢查是否整除
patch_size = 8
img_size_list = [128, 16, 8]

for img_size in img_size_list:
    assert img_size % patch_size == 0, f"{img_size} not divisible by {patch_size}"
    print(f"✓ {img_size} / {patch_size} = {img_size // patch_size}")
```

### 步驟 3: 調整數據設置

```yaml
data:
  crop_length: 512  # 確保能被 patch_size 整除
  # 512 % 8 = 0 ✓
```

### 步驟 4: 更新預期性能

```yaml
training:
  # patch_size=8 可以用更大的 batch_size
  train_batch_size: 32  # 從 16 增加到 32
  accumulate_grad_batches: 8  # 可以減少
```

---

## 📈 性能基準測試 (估算)

基於 128×512 piano roll, batch_size=16:

| patch_size | 序列長度 | 訓練速度 | GPU 記憶體 | 推薦 GPU |
|-----------|---------|---------|-----------|---------|
| 2 | 16,384 | 0.06x | 40+ GB | ❌ 不可行 |
| **4** | **4,096** | **1.0x** | **20-24 GB** | **24GB+ (RTX 3090/4090)** |
| 8 | 1,024 | 16x | 8-12 GB | ✅ 16GB (RTX 4060 Ti) |
| 16 | 256 | 256x | 4-6 GB | ✅ 8GB (任何現代 GPU) |

---

## 🎯 選擇建議

### 推薦使用 patch_size=4 如果:
- ✅ 你有 24GB+ GPU (RTX 3090, 4090, A5000, A6000)
- ✅ 想要高質量的音樂生成
- ✅ 關注和弦和旋律細節
- ✅ 訓練時間可接受

### 考慮使用 patch_size=8 如果:
- ⚠️ GPU 記憶體有限 (12-16GB)
- ⚠️ 需要快速實驗迭代
- ⚠️ 只關注整體結構
- ⚠️ 生成慢速/環境音樂

### 不推薦 patch_size=2:
- ❌ 計算成本極高
- ❌ 收益有限
- ❌ 實用性低

### 不推薦 patch_size=16:
- ❌ 分辨率太低
- ❌ 音樂質量差
- ❌ 僅用於測試

---

## 🔍 實際例子

### 音符表達能力

```
C major chord (C4, E4, G4) = MIDI notes 60, 64, 67

patch_size=2: (每 patch = 2 semitones)
  → Patch boundaries: [60-61], [62-63], [64-65], [66-67]
  → C4, E4, G4 分別在不同 patches ✓ 清晰

patch_size=4: (每 patch = 4 semitones)
  → Patch boundaries: [60-63], [64-67], ...
  → C4 和 E4 可能在不同 patches ✓ 較清晰
  → E4 和 G4 在同一 patch ⚠️ 略模糊

patch_size=8: (每 patch = 8 semitones)
  → Patch boundaries: [60-67], [68-75], ...
  → C4, E4, G4 全在同一 patch ❌ 模糊

結論: patch_size=4 對大部分和弦仍有良好表達
```

### 快速音符

```
16th note runs (每音符 = 1 時間步)

patch_size=2:
  → 每 patch = 2 steps
  → 可以捕捉 32nd notes ✓

patch_size=4:
  → 每 patch = 4 steps  
  → 可以捕捉 16th notes ✓

patch_size=8:
  → 每 patch = 8 steps
  → 最快只能捕捉 8th notes ⚠️

patch_size=16:
  → 每 patch = 16 steps
  → 最快只能捕捉 quarter notes ❌

結論: patch_size=4 適合大部分速度的音樂
```

---

## 📊 總結表格

| 指標 | patch_size=2 | **patch_size=4** | patch_size=8 | patch_size=16 |
|-----|-------------|-----------------|-------------|--------------|
| **序列長度** | 16,384 | **4,096** | 1,024 | 256 |
| **計算速度** | 0.06x | **1.0x** | 16x | 256x |
| **GPU 記憶體** | 40+ GB | **20-24 GB** | 8-12 GB | 4-6 GB |
| **Pitch 分辨率** | 極高 | **高** | 中 | 低 |
| **Time 分辨率** | 極高 | **高** | 中 | 低 |
| **音樂質量** | 最佳 | **優秀** | 可接受 | 差 |
| **訓練時間** | 極長 | **長** | 中 | 短 |
| **實用性** | 低 | **高** ⭐ | 中 | 低 |

---

## 💡 最佳實踐

1. **默認使用 patch_size=4**
   - 經過驗證的平衡選擇
   - 與當前架構完美匹配

2. **GPU 記憶體有限時使用 patch_size=8**
   - 同時調整 img_size_list
   - 增加 batch_size 以補償

3. **不要使用極端值**
   - patch_size=2: 過於昂貴
   - patch_size=16: 質量太差

4. **更改時同時調整**:
   - ✅ img_size_list
   - ✅ embed_dim_list  
   - ✅ num_blocks_list
   - ✅ num_heads_list
   - ✅ batch_size
   - ✅ crop_length 整除性

5. **驗證配置**:
   ```python
   # 確保所有尺寸可以被 patch_size 整除
   assert piano_roll_height % patch_size == 0
   assert crop_length % patch_size == 0
   for img_size in img_size_list:
       assert img_size % patch_size == 0
   ```

---

## 🔗 相關文檔

- `docs/CROP_LENGTH_IMPACT.md`: Crop length 影響分析
- `docs/PIANO_ROLL_SIZES.md`: Piano roll 尺寸指南
- `models/model_config.py`: 配置系統
- `REFACTORING_COMPLETE.md`: 配置系統重構

---

**結論**: `patch_size=4` 是經過深思熟慮的默認選擇，在音樂質量、計算效率和硬體需求之間達到了最佳平衡。除非有特殊需求（如硬體受限或快速實驗），否則不建議更改。

