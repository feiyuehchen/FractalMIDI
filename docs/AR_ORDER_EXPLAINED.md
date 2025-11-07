# Autoregressive (AR) 生成順序詳解

## 核心問題

**AR 的順序是怎麼來的？**

答案：**固定的、預定義的順序**（從左到右、從上到下），不是隨機的！

## 關鍵代碼

### AR 的生成循環

```python
# model.py 第 936 行
for patch_idx in range(self.seq_len):
    # 按照 0, 1, 2, 3, ..., seq_len-1 的順序生成
    conds = self.predict(canvas, cond_list)
    cond_for_patch = [c[:, patch_idx] for c in conds]
    patch_content = next_level_sample_function(cond_list=cond_for_patch, ...)
    canvas[:, patch_idx] = patch_flat
```

**關鍵點**：
- `for patch_idx in range(self.seq_len)` → **順序遍歷**
- 沒有隨機打亂
- 沒有採樣順序
- 就是 0, 1, 2, 3, ..., 127

## 順序的空間映射

### Level 0 的例子（128x128 → 8x16 patches）

```python
# model.py 第 1191-1194 行
h_patches = 128 // img_size_list[fractal_level+1]  # 128 // 16 = 8
max_w_patches = 256 // img_size_list[fractal_level+1]  # 256 // 16 = 16
expected_seq_len = h_patches * max_w_patches  # 8 * 16 = 128
```

**Patch 的排列方式**：

```
Piano Roll: 128 (高度) x 256 (寬度)
分成: 8 (行) x 16 (列) 個 patches

Patch 編號順序（row-major order）:
┌────────────────────────────────────────┐
│  0   1   2   3   4   5   6   7   8   9 │ ← 第 1 行
│ 10  11  12  13  14  15  16  17  18  19 │ ← 第 2 行
│ 20  21  22  23  24  25  26  27  28  29 │ ← 第 3 行
│ 30  31  32  33  34  35  36  37  38  39 │ ← 第 4 行
│ 40  41  42  43  44  45  46  47  48  49 │ ← 第 5 行
│ 50  51  52  53  54  55  56  57  58  59 │ ← 第 6 行
│ 60  61  62  63  64  65  66  67  68  69 │ ← 第 7 行
│ 70  71  72  73  74  75  76  77  78  79 │ ← 第 8 行
│ ... (繼續到 127)
└────────────────────────────────────────┘

生成順序: 0 → 1 → 2 → 3 → ... → 127
```

### 為什麼是這個順序？

**Row-major order（行優先順序）**：
- 先填充第一行的所有 patches
- 再填充第二行的所有 patches
- 依此類推

**對應到音樂**：
```
時間 →
┌─────────────────────────────────┐
│ ████████████████████████████████ │ ← 高音（C8-B7）
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │ ← 中音（C4-B3）
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │ ← 低音（C0-B-1）
└─────────────────────────────────┘

生成順序: 從左到右、從上到下
```

## 與 MAR 的對比

### AR（Autoregressive）

```python
# 固定順序
for patch_idx in range(128):  # 0, 1, 2, ..., 127
    生成 patch[patch_idx]
```

**特點**：
- ✅ 順序固定（0 → 1 → 2 → ...）
- ✅ 每個 patch 依賴之前所有的 patches
- ✅ 嚴格的因果關係（causal）
- ❌ 必須序列生成，無法並行

### MAR（Masked Autoregressive）

```python
# model.py 第 264-267 行
def sample_orders(self, bsz, actual_seq_len, device='cuda'):
    """Sample random orders for masking."""
    orders = torch.argsort(torch.rand(bsz, actual_seq_len, device=device), dim=1).long()
    return orders
```

**特點**：
- ✅ 順序隨機（每次運行不同）
- ✅ 可以並行生成多個 patches
- ✅ 迭代細化（從粗到細）
- ✅ 更快的生成速度

**MAR 的隨機順序例子**：
```
第一次運行: [42, 7, 91, 15, 63, ...]
第二次運行: [88, 3, 56, 120, 9, ...]
第三次運行: [17, 99, 4, 71, 38, ...]
```

## AR 順序的優缺點

### ✅ 優點

#### 1. 時間連貫性
```
生成順序 = 時間順序
Patch 0 (時間 0-15) → Patch 1 (時間 16-31) → Patch 2 (時間 32-47) → ...
```
- 自然地保持音樂的時間流動
- 前面的音符影響後面的音符
- 符合音樂創作的直覺

#### 2. 因果關係明確
```
Patch N 只依賴 Patch 0, 1, 2, ..., N-1
不會依賴未來的 patches
```
- 符合真實的音樂創作過程
- 訓練和推論一致
- 理論上更容易學習長期依賴

#### 3. 實現簡單
```python
for i in range(seq_len):
    generate(i)
```
- 代碼直觀
- 容易理解和調試
- 不需要額外的順序採樣邏輯

### ❌ 缺點

#### 1. 生成速度慢
```
必須等待 Patch 0 完成 → 才能生成 Patch 1
必須等待 Patch 1 完成 → 才能生成 Patch 2
...
```
- 無法並行化
- 生成時間 = seq_len × 單個 patch 時間
- Level 0: 128 個 patches 需要 128 次調用

#### 2. 錯誤累積
```
Patch 0 錯誤 → 影響 Patch 1 → 影響 Patch 2 → ...
```
- 早期的錯誤會傳播
- 無法修正之前的錯誤
- 可能導致後期生成質量下降

#### 3. 缺乏全局視野
```
生成 Patch 50 時，只看到 Patch 0-49
不知道 Patch 51-127 會是什麼
```
- 難以規劃整體結構
- 可能導致音樂缺乏整體性
- 無法「回頭修改」

## 為什麼不隨機化 AR 的順序？

### 理論上可以
```python
# 可以這樣做（但我們沒有）
order = torch.randperm(self.seq_len)
for i in range(self.seq_len):
    patch_idx = order[i]
    generate(patch_idx)
```

### 但這樣做的問題

#### 1. 破壞時間連貫性
```
隨機順序: [42, 7, 91, 15, ...]
生成時間 42 → 時間 7 → 時間 91 → 時間 15
```
- 音樂在時間上跳來跳去
- 難以學習旋律和節奏
- 不符合音樂創作的自然流程

#### 2. 訓練推論不一致
```
訓練時: 順序 0 → 1 → 2 → ...
推論時: 隨機順序
```
- 模型沒有見過隨機順序
- 可能產生奇怪的結果
- 需要特殊的訓練策略

#### 3. 失去 AR 的優勢
```
AR 的核心優勢 = 明確的因果關係
隨機順序 = 破壞因果關係
```
- 變得更像 MAR
- 但沒有 MAR 的並行優勢
- 兩頭不討好

## 實際生成過程示例

### Level 0 (AR, 128 patches)

```
Step 0: 初始化
Canvas: [-1, -1, -1, -1, ..., -1]  (全白)

Step 1: 生成 Patch 0
Canvas: [X, -1, -1, -1, ..., -1]
        ↑ 剛生成的

Step 2: 生成 Patch 1（基於 Patch 0）
Canvas: [X, Y, -1, -1, ..., -1]
        ↑  ↑
        已有 剛生成的

Step 3: 生成 Patch 2（基於 Patch 0, 1）
Canvas: [X, Y, Z, -1, ..., -1]
        ↑  ↑  ↑
        已有 已有 剛生成的

...

Step 128: 生成 Patch 127（基於所有之前的）
Canvas: [X, Y, Z, ..., W]  (全部填滿)
```

### Condition 的演變

```python
# Step 1
conds = self.predict(canvas, cond_list)
# canvas = [-1, -1, -1, ...]
# conds[0][:, 0] = 基於全白 canvas 的 condition

# Step 2
conds = self.predict(canvas, cond_list)
# canvas = [X, -1, -1, ...]
# conds[0][:, 1] = 基於 [X, -1, -1, ...] 的 condition

# Step 3
conds = self.predict(canvas, cond_list)
# canvas = [X, Y, -1, ...]
# conds[0][:, 2] = 基於 [X, Y, -1, ...] 的 condition
```

**關鍵理解**：
- 每次 `predict()` 都看到**當前的整個 canvas**
- 但只提取**當前 patch 的 condition**：`c[:, patch_idx]`
- Condition 反映了「已生成內容」的影響

## 空間上的因果關係

### Causal Attention 的作用

```python
# model.py 第 755-757 行
class CausalAttention(nn.Module):
    """Causal self-attention for autoregressive generation."""
```

**Causal Mask**：
```
Patch 0 可以看到: [Patch 0]
Patch 1 可以看到: [Patch 0, Patch 1]
Patch 2 可以看到: [Patch 0, Patch 1, Patch 2]
...
Patch N 可以看到: [Patch 0, 1, 2, ..., N]
```

**實現方式**：
```python
# 創建 causal mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# mask[i, j] = 1 表示 position i 不能看到 position j

例如 seq_len=4:
mask = [[0, 1, 1, 1],   ← Patch 0 只能看到自己
        [0, 0, 1, 1],   ← Patch 1 可以看到 0, 1
        [0, 0, 0, 1],   ← Patch 2 可以看到 0, 1, 2
        [0, 0, 0, 0]]   ← Patch 3 可以看到 0, 1, 2, 3
```

## 與原始 FractalGen 的比較

### 原始 FractalGen（圖像生成）

```python
# 2D 空間的 raster scan order
for h in range(H):
    for w in range(W):
        generate(h, w)
```

**順序**：
```
0  1  2  3  4
5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
```

### 我們的實現（Piano Roll）

```python
# 1D 序列（但映射到 2D 空間）
for idx in range(seq_len):
    generate(idx)
```

**順序**：
```
相同！也是 row-major order
0  1  2  3  4  ...
16 17 18 19 20 ...
32 33 34 35 36 ...
...
```

## 總結

### AR 順序的本質

1. **固定的、預定義的**
   - 不是隨機的
   - 不是採樣的
   - 就是 0, 1, 2, 3, ..., seq_len-1

2. **Row-major order**
   - 從左到右
   - 從上到下
   - 符合時間流動

3. **嚴格的因果關係**
   - Patch N 只依賴 Patch 0~N-1
   - 通過 Causal Attention 實現
   - 訓練和推論一致

### 與 MAR 的關鍵區別

| 特性 | AR | MAR |
|------|----|----|
| 順序 | 固定（0→1→2→...） | 隨機（每次不同） |
| 並行 | ❌ 必須序列 | ✅ 可以並行 |
| 速度 | 慢 | 快 |
| 因果 | 嚴格 | 靈活 |
| 實現 | 簡單 | 複雜 |

### 為什麼這樣設計？

1. **符合音樂的時間性**
   - 音樂是時間藝術
   - 從左到右 = 從過去到未來
   - 自然的創作流程

2. **理論上的優勢**
   - 明確的因果關係
   - 容易學習長期依賴
   - 訓練穩定

3. **實現簡單**
   - 不需要複雜的順序採樣
   - 代碼直觀易懂
   - 容易調試

**最重要的理解**：
> AR 的順序不是「來自」某個地方，而是**設計決策**。
> 我們選擇固定的、從左到右的順序，因為這最符合音樂的本質。

---

**文檔版本**: 1.0  
**日期**: 2025-11-07  
**作者**: AI Assistant

