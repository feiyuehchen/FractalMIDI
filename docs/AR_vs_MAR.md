# AR vs MAR Implementation in FractalGen

## 問題回顧

之前的 AR 實作無法正常運作，原因是**維度不匹配**和**不是真正的 autoregressive generation**。

## 根本差異

### MAR (Masked Autoregressive)
- **並行生成**：一次生成所有 patches
- **使用 masking**：用 mask token 標記要生成的位置
- **Batching 友好**：可以將所有 patches 當作一個 batch 一起處理
- **Forward pass**：
  ```
  Level 0: (bsz, seq_len, patch_size^2) → mask some → predict all
  Level 1: (bsz*seq_len, patch_size^2) → all patches in one batch
  ```

### AR (Autoregressive)
- **序列生成**：一次生成一個 patch
- **使用 causal masking**：用已生成的 patches 作為 context
- **必須 sequential**：每個 patch 依賴前面的 patches
- **Forward pass**：
  ```
  Level 0: for each patch in seq_len:
              cond = predict(canvas[:, :patch_idx+1])
              patch = next_level.sample(cond)
              canvas[:, patch_idx] = patch
  ```

## 之前的錯誤實作

```python
# ❌ 錯誤：試圖一次生成所有 patches
zero_patches = torch.zeros(bsz, seq_len, patch_size^2)
conds = self.predict(zero_patches, cond_list)
conds_flat = conds.reshape(bsz * seq_len, -1)  # Flatten all patches

# 這裡的問題：下一層返回 (bsz, 1, img_size, img_size)
# 但我們期望 (bsz*seq_len, 1, patch_size, patch_size)
sampled = next_level_sample_function(conds_flat)
```

**問題**：
1. 沒有真正的 sequential generation
2. Batching 維度不匹配
3. 下層返回完整圖像，上層期望 patch

## 新的正確實作

```python
# ✓ 正確：逐個生成 patches
canvas = torch.zeros(bsz, seq_len, patch_size^2)

for patch_idx in range(seq_len):
    # 使用當前 canvas 狀態預測條件
    conds = self.predict(canvas, cond_list)
    
    # 只取當前 patch 的條件：(bsz, seq_len, dim) → (bsz, dim)
    cond_for_patch = [c[:, patch_idx] for c in conds]
    
    # 為單個 patch 生成內容
    patch_content = next_level_sample_function(
        cond_list=cond_for_patch,
        cfg=cfg,
        temperature=temperature,
        filter_threshold=filter_threshold
    )
    # patch_content: (bsz, 1, patch_size, patch_size)
    
    # 展平並更新 canvas
    canvas[:, patch_idx] = patch_content.reshape(bsz, -1)

# 最後將 canvas 轉換為圖像
return unpatchify(canvas)
```

## 維度流程對比

### MAR 的維度流程
```
Level 0 (bsz=2, seq_len=16):
  Input: (2, 16, 64) with some masked
  ↓
  Predict: (2, 16, embed_dim) for all patches
  ↓
  Flatten: (32, embed_dim) → send to Level 1 as batch
  ↓
Level 1 receives: (32, embed_dim)
  Generates: (32, 1, 8, 8) for all 32 patches at once
  ↓
Level 0 receives: (32, 1, 8, 8)
  Reshape: (2, 16, 64)
  ✓ 維度匹配！
```

### AR 的維度流程
```
Level 0 (bsz=2, seq_len=16):
  Canvas: (2, 16, 64) initialized with zeros
  ↓
  For patch_idx in 0..15:
    Predict: (2, 16, embed_dim)
    ↓
    Extract: (2, embed_dim) for patch_idx
    ↓
  Level 1 receives: (2, embed_dim) for ONE patch
    Generates: (2, 1, 8, 8) for this single patch
    ↓
  Level 0 receives: (2, 1, 8, 8)
    Flatten: (2, 64)
    Update: canvas[:, patch_idx] = (2, 64)
  ✓ 維度匹配！
```

## 性能考量

### MAR 優點
- ✓ **快速**：所有 patches 並行生成
- ✓ **GPU 利用率高**：大 batch 充分利用並行計算
- ✓ **訓練穩定**：gradients 來自所有 patches

### AR 優點
- ✓ **更好的序列建模**：每個 patch 看到之前的 context
- ✓ **更符合音樂生成**：時間上的依賴關係
- ✓ **理論上更強的表達能力**

### AR 缺點
- ✗ **慢**：必須 sequential，無法完全並行
- ✗ **記憶體消耗**：需要保存整個 canvas 的 gradient
- ✗ **推理時間長**：seq_len 次 forward pass

## 混合策略

最佳實踐可能是**混合使用**：

```python
# 範例：頂層用 AR 建模長期依賴，底層用 MAR 加速
generator_type_list = ["ar", "ar", "mar", "mar"]
```

- **Level 0, 1 (AR)**：建模音樂的時間結構
- **Level 2, 3 (MAR)**：並行生成細節（velocity, pitch）

## 訓練時的差異

### AR Training
- Forward pass 照常：所有 patches 一起輸入
- Loss 計算：所有位置的 reconstruction loss
- 不需要 masking
- Causal attention 確保位置 i 只看到位置 0..i-1

### MAR Training
- Forward pass：隨機 mask 一些 patches
- Loss 計算：只計算被 masked 位置的 loss
- Mask ratio 動態調整（truncated normal）

## 現在可以訓練了！

修改後的 AR 實作：
1. ✓ 真正的 sequential generation
2. ✓ 維度匹配
3. ✓ 與 hierarchical sampling 兼容
4. ✓ 正確處理 CFG

你可以開始訓練 `ar,ar,ar,ar` 配置，應該會正常運作！

