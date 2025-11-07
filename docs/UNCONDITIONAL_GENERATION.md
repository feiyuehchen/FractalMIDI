# Unconditional 生成流程詳解

## 概述

Unconditional generation 是指**不依賴任何輸入條件**（如前綴、類別標籤等）來生成全新的 piano roll。模型從純粹的隨機性和學習到的音樂結構開始生成。

## 完整流程圖

```
用戶調用
    ↓
inference.py: model.sample(batch_size=1, cond_list=None, ...)
    ↓
PianoRollFractalGen.sample()
    ├─ cond_list=None → 使用 dummy_cond
    ├─ 初始化白色 canvas: torch.full(..., -1.0)
    └─ 開始遞歸生成
        ↓
    Level 0 (AR/MAR): 128x128 → 生成 8x16 patches
        ├─ 初始化: canvas 全部 -1 (白色)
        ├─ 每個 patch 調用 Level 1
        └─ 逐步填充 canvas
            ↓
        Level 1 (AR/MAR): 16x16 → 生成每個 patch
            ├─ 接收來自 Level 0 的 condition
            ├─ 每個 patch 調用 Level 2
            └─ 返回 16x16 的內容
                ↓
            Level 2 (AR/MAR): 4x4 → 生成每個 patch
                ├─ 接收來自 Level 1 的 condition
                ├─ 每個 patch 調用 Level 3
                └─ 返回 4x4 的內容
                    ↓
                Level 3 (VelocityLoss): 1x1 → 生成單個 velocity
                    ├─ 接收來自 Level 2 的 condition
                    ├─ 從 256 個 velocity 值中採樣
                    └─ 返回 [-1, 1] 的值
                        ↓
                    逐層返回並組合
                        ↓
                最終輸出: (1, 1, 128, 256) piano roll
```

## 詳細步驟

### 1. 入口點：inference.py

```python
# inference.py 第 261 行
generated = model.model.sample(
    batch_size=1,           # 生成 1 個樣本
    cond_list=None,         # ← 關鍵：None 表示 unconditional
    num_iter_list=[8, 4, 2, 1],  # 每層的迭代次數
    cfg=1.0,                # Classifier-free guidance (1.0 = 不使用)
    cfg_schedule='constant',
    temperature=1.0,        # 採樣溫度
    filter_threshold=0.0,
    fractal_level=0         # 從頂層開始
)
```

**關鍵參數**：
- `cond_list=None` → 觸發 unconditional 模式
- `batch_size=1` → 生成 1 個樣本
- `num_iter_list` → 控制每層的生成質量

### 2. Dummy Condition 初始化

```python
# model.py 第 1273-1276 行
if cond_list is None:
    # Use dummy condition
    dummy_embedding = self.dummy_cond.expand(batch_size, -1)
    cond_list = [dummy_embedding for _ in range(5)]
```

**`dummy_cond` 是什麼？**
- 一個**可學習的參數**：`nn.Parameter(torch.zeros(1, embed_dim))`
- 在訓練時學習「無條件生成」的表示
- 初始化為 0，通過訓練學習到合適的值
- 相當於告訴模型：「請生成一個典型的音樂片段」

**為什麼需要 5 個？**
```python
cond_list = [dummy_embedding for _ in range(5)]
```
- 模型使用 5 個 condition 輸入（中間、上、右、下、左）
- 全部使用相同的 `dummy_embedding`
- 表示「沒有任何空間上的條件」

### 3. Canvas 初始化

```python
# model.py 第 1279-1284 行
if return_intermediates and fractal_level == 0 and _intermediates_list is None:
    _intermediates_list = {
        'frames': [],
        'canvas': torch.full((batch_size, 1, 128, 256), -1.0)
    }
```

**初始狀態**：
- 全部填充 -1.0（白色/靜音）
- 形狀：(1, 1, 128, 256)
  - 1 個樣本
  - 1 個通道（velocity）
  - 128 個音高
  - 256 個時間步

### 4. Level 0：頂層生成（AR 或 MAR）

#### 如果是 AR（Autoregressive）：

```python
# model.py 第 921-930 行
# Initialize canvas with -1 (white/silence)
canvas = torch.full(
    (bsz, self.seq_len, self.patch_size**2),
    -1.0,
    device=cond_list[0].device,
    dtype=cond_list[0].dtype
)
```

**AR 的生成過程**：
```
seq_len = 128 (8x16 patches)

for patch_idx in range(128):  # 逐個生成
    1. 使用當前 canvas 預測下一個 patch 的 condition
       conds = self.predict(canvas, cond_list)
    
    2. 提取當前 patch 的 condition
       cond_for_patch = [c[:, patch_idx] for c in conds]
    
    3. 調用下一層生成這個 patch
       patch_content = next_level_sample_function(
           cond_list=cond_for_patch,
           ...
       )
    
    4. 更新 canvas
       canvas[:, patch_idx] = patch_flat
    
    5. 記錄中間狀態（每 16 patches）
```

**關鍵特點**：
- **序列生成**：每個 patch 依賴之前生成的 patches
- **自回歸**：使用已生成的內容作為上下文
- **逐步建構**：從左到右、從上到下填充

#### 如果是 MAR（Masked Autoregressive）：

```python
# model.py 第 473-476 行
mask = torch.ones(bsz, actual_seq_len, device=cond_list[0].device)
patches = torch.full((bsz, actual_seq_len, 1 * self.patch_size**2), -1.0, device=cond_list[0].device)
orders = self.sample_orders(bsz, actual_seq_len, device=cond_list[0].device)
```

**MAR 的生成過程**：
```
num_iter = 8  # Level 0 的迭代次數

for step in range(8):
    1. 決定這次要生成哪些 patches
       mask_ratio = cos(π/2 * (step+1) / 8)
       # step 0: 100% masked
       # step 1: 92% masked
       # ...
       # step 7: 8% masked
    
    2. 預測被 mask 的 patches 的 condition
       cond_list_next = self.predict(patches, mask, cond_list)
    
    3. 對每個被 mask 的 patch 調用下一層
       for each masked patch:
           patch_content = next_level_sample_function(...)
           patches[patch_idx] = patch_content
    
    4. 更新 mask（減少 masked patches）
```

**關鍵特點**：
- **並行生成**：同時生成多個 patches
- **迭代細化**：每次迭代填充更多細節
- **從粗到細**：先生成整體結構，再填充細節

### 5. Level 1-2：中間層生成

每一層都重複類似的過程：

```python
# model.py 第 1288-1302 行
def next_level_sample_function(cond_list, cfg, temperature, filter_threshold, patch_pos=None):
    result = self.next_fractal.sample(
        batch_size=cond_list[0].size(0) if cfg == 1.0 else cond_list[0].size(0) // 2,
        cond_list=cond_list,  # ← 來自上一層的 condition
        num_iter_list=num_iter_list,
        cfg=cfg,
        cfg_schedule="constant",
        temperature=temperature,
        filter_threshold=filter_threshold,
        fractal_level=fractal_level + 1,  # ← 遞增層級
        return_intermediates=return_intermediates,
        _intermediates_list=_intermediates_list,
        _patch_pos=patch_pos
    )
    return result
```

**層級結構**：
```
Level 0: 128x128 → 8x16 = 128 patches
    ↓ 每個 patch 調用 Level 1
Level 1: 16x16 → 4x4 = 16 patches
    ↓ 每個 patch 調用 Level 2
Level 2: 4x4 → 4x1 = 4 patches
    ↓ 每個 patch 調用 Level 3
Level 3: 1x1 → 單個 velocity 值
```

### 6. Level 3：Velocity 採樣

```python
# model.py 第 1066-1145 行
def sample(self, cond_list, temperature, cfg, ...):
    # 1. 初始化為 -1 (白色/靜音)
    velocity_values = torch.full((bsz, 1), -1.0, device=cond_list[0].device)
    
    # 2. 轉換為 [0, 1] 用於預測
    velocity_for_pred = (velocity_values + 1.0) / 2.0
    
    # 3. 預測 logits（256 個 velocity 值的概率）
    logits, _ = self.predict(velocity_for_pred, cond_list)
    
    # 4. 應用溫度
    logits = logits / max(temperature, 1e-8)
    
    # 5. 數值穩定性
    logits = torch.clamp(logits, min=-20, max=20)
    
    # 6. 轉換為概率
    probs = torch.softmax(logits, dim=-1)
    probs = probs + 1e-10
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # 7. 採樣
    sampled_ids = torch.multinomial(probs, num_samples=1).reshape(-1)
    
    # 8. 轉換回 [-1, 1] 範圍
    velocity_values[:, 0] = (sampled_ids.float() / 255.0) * 2.0 - 1.0
    
    # 9. 返回 (bsz, 1, 1, 1)
    return velocity_values.view(bsz, 1, 1, 1)
```

**採樣過程**：
1. 從 condition 預測 256 個可能的 velocity 值的概率分佈
2. 使用溫度控制隨機性（temperature=1.0 表示標準採樣）
3. 從分佈中採樣一個 velocity 值
4. 轉換為 [-1, 1] 範圍

### 7. 逐層返回和組合

```
Level 3 返回: (1, 1, 1, 1) - 單個 velocity
    ↓ 組合成 4x1
Level 2 返回: (1, 1, 4, 4) - 4x4 patch
    ↓ 組合成 4x4
Level 1 返回: (1, 1, 16, 16) - 16x16 patch
    ↓ 組合成 8x16
Level 0 返回: (1, 1, 128, 256) - 完整 piano roll
```

## Condition 的流動

### Unconditional 的 Condition 鏈

```
初始: dummy_cond (可學習參數)
    ↓
Level 0: [dummy_cond, dummy_cond, dummy_cond, dummy_cond, dummy_cond]
    ↓ predict() 生成新的 conditions
Level 1: [cond_0, cond_1, cond_2, cond_3, cond_4]
    ↓ 基於 Level 0 的 canvas
Level 2: [cond_0, cond_1, cond_2, cond_3, cond_4]
    ↓ 基於 Level 1 的輸出
Level 3: [cond_0, cond_1, cond_2, cond_3, cond_4]
    ↓ 基於 Level 2 的輸出
採樣 velocity
```

**關鍵理解**：
- 雖然是 "unconditional"，但每層仍然有 condition
- 這些 condition 來自**上一層的輸出**，不是外部輸入
- `dummy_cond` 只在最頂層使用，之後都是模型內部生成的 condition

## 隨機性來源

### 1. Velocity 採樣
```python
sampled_ids = torch.multinomial(probs, num_samples=1)
```
- 從概率分佈中隨機採樣
- 每次運行結果不同

### 2. MAR 的 Order 採樣
```python
orders = torch.argsort(torch.rand(bsz, actual_seq_len, device=device), dim=1)
```
- 隨機決定 patches 的生成順序
- 增加生成的多樣性

### 3. 溫度控制
```python
logits = logits / temperature
```
- temperature > 1.0：更隨機（更多樣性）
- temperature < 1.0：更確定（更保守）
- temperature = 1.0：標準採樣

## 為什麼能生成音樂？

### 1. 訓練時學習的知識

模型在訓練時看到大量真實的 piano rolls：
```python
# 訓練時
loss, stats = model(real_piano_roll, cond_list=None)
```

- 學習音樂的統計規律
- 學習和弦、旋律、節奏模式
- 學習音符之間的關係
- `dummy_cond` 學習「典型音樂」的表示

### 2. 階層式結構

```
Level 0 (128x128): 學習整體結構
    - 樂句的長度
    - 音樂的密度
    - 整體的動態變化

Level 1 (16x16): 學習局部模式
    - 和弦進行
    - 短旋律片段
    - 節奏型

Level 2 (4x4): 學習細節
    - 音符組合
    - 裝飾音
    - 微小的時間變化

Level 3 (1x1): 學習 velocity
    - 力度變化
    - 表情
    - 強弱對比
```

### 3. 自回歸的連貫性

AR 模式確保生成的連貫性：
```
生成 patch_1 → 基於 patch_1 生成 patch_2 → 基於 patch_1,2 生成 patch_3 → ...
```

這樣生成的音樂在時間上是連貫的。

## 與 Conditional 的區別

### Unconditional
```python
cond_list = None
    ↓
dummy_embedding = self.dummy_cond.expand(batch_size, -1)
cond_list = [dummy_embedding for _ in range(5)]
```
- 從「典型音樂」的表示開始
- 完全由模型決定生成什麼
- 每次運行產生不同的音樂

### Conditional（例如：prefix-based）
```python
# 提供前 64 步作為條件
prefix = real_piano_roll[:, :, :, :64]
cond_list = model.encode(prefix)  # 編碼前綴
    ↓
生成後續內容
```
- 從給定的音樂片段繼續
- 延續前綴的風格和內容
- 生成是前綴的「合理延續」

## 實際例子

### 生成過程可視化

```
初始狀態（全白）:
████████████████████████████████  (128x256, 全部 -1)

Level 0, Iteration 1 (AR: patch 1/128):
█▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (開始填充第一個 patch)

Level 0, Iteration 16 (AR: patch 16/128):
█▓▒░▓▒░▓░░░░░░░░░░░░░░░░░░░░░░░  (填充了 1/8)

Level 0, Iteration 64 (AR: patch 64/128):
█▓▒░▓▒░▓▒░▓▒░▓▒░▓░░░░░░░░░░░░░░  (填充了一半)

Level 0, Iteration 128 (AR: patch 128/128):
█▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░  (完成！)
```

### 實際輸出統計

```
生成結果:
  Shape: (1, 1, 128, 256)
  Range: [-1.000, 1.000]
  Mean: -0.897
  
值分佈:
  < -0.5 (靜音): 93.18%  ← 大部分是背景
  [-0.5, 0.0):   1.31%   ← 輕音符
  [0.0, 0.5):    1.90%   ← 中等音符
  >= 0.5 (響亮): 3.61%   ← 響亮音符
```

## 總結

**Unconditional 生成的本質**：
1. 從 `dummy_cond`（學習到的「典型音樂」表示）開始
2. 初始化全白 canvas（-1 = 靜音）
3. 階層式地、逐步地填充內容
4. 每層生成都基於上一層的輸出
5. 最底層通過隨機採樣引入多樣性
6. 最終產生完整的、連貫的 piano roll

**關鍵要點**：
- ✅ "Unconditional" 不代表「沒有條件」，而是「沒有外部輸入」
- ✅ 模型內部仍然使用條件（來自上一層）
- ✅ `dummy_cond` 是可學習的，代表「典型音樂」
- ✅ 隨機性來自底層的採樣過程
- ✅ 階層式結構確保從整體到細節的連貫性

---

**文檔版本**: 1.0  
**日期**: 2025-11-07  
**作者**: AI Assistant

