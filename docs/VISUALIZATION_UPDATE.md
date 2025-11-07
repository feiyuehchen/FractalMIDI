# 可視化更新：支援 [-1, 1] 範圍

## 更新日期
2025-11-07

## 問題
模型現在輸出 [-1, 1] 範圍的值（-1 = 白色/靜音，1 = 有色/響亮），但可視化函數假設輸入是 [0, 1] 範圍。

## 解決方案

### 修改的檔案
- `visualizer.py` - 所有可視化函數

### 修改內容

#### 1. `piano_roll_to_image()`
```python
# 自動檢測並轉換範圍
if piano_roll_np.min() < -0.5:  # Likely in [-1, 1] range
    # Convert from [-1, 1] to [0, 1]
    # -1 (silence/white) -> 0, 1 (loud) -> 1
    piano_roll_np = (piano_roll_np + 1.0) / 2.0
    piano_roll_np = np.clip(piano_roll_np, 0, 1)
```

**特點**：
- ✅ 自動檢測輸入範圍
- ✅ 支援 [-1, 1] 和 [0, 1] 兩種格式
- ✅ 向後兼容舊代碼

#### 2. `visualize_piano_roll()`
```python
# Normalize to [0, 1] if in [-1, 1] range
if piano_roll_np.min() < -0.5:
    piano_roll_np = (piano_roll_np + 1.0) / 2.0
    piano_roll_np = np.clip(piano_roll_np, 0, 1)
```

#### 3. `compare_piano_rolls()`
```python
# Normalize both piano rolls
if original.min() < -0.5:
    original = (original + 1.0) / 2.0
    original = np.clip(original, 0, 1)
if generated.min() < -0.5:
    generated = (generated + 1.0) / 2.0
    generated = np.clip(generated, 0, 1)
```

## 值範圍對應

### 模型輸出 → 可視化
```
模型輸出範圍: [-1, 1]
  -1.0 → 靜音/白色背景 → colormap 的 0.0
  -0.5 → 很輕的音符    → colormap 的 0.25
   0.0 → 中等音符      → colormap 的 0.5
   0.5 → 響亮音符      → colormap 的 0.75
   1.0 → 最大音量      → colormap 的 1.0
```

### Colormap 顏色
```
0.0 (靜音):     黑色
0.05-0.2:       藍色
0.2-0.35:       青色
0.35-0.5:       綠色
0.5-0.65:       黃綠色
0.65-0.8:       黃橙色
0.8-0.9:        橙色
0.9-1.0:        紅色
```

## 測試結果

### 測試 1: [-1, 1] 範圍
```
輸入範圍: [-1.000, 1.000]
輸出圖片: 512x256 RGB
背景像素: 99.22% (近黑色 = 靜音)
音符像素: 0.78% (有色 = 有音符)
✅ 正確顯示
```

### 測試 2: [0, 1] 範圍（向後兼容）
```
輸入範圍: [0.000, 1.000]
輸出圖片: 512x256 RGB
背景像素: 99.22%
音符像素: 0.78%
✅ 與 [-1, 1] 結果相同
```

### 測試 3: 實際模型生成
```
輸入範圍: [-1.000, 1.000]
輸出圖片: 512x256 RGB
背景像素: 93.14%
音符像素: 6.81%
✅ 正確顯示生成結果
```

## 使用方式

### 基本使用
```python
from visualizer import piano_roll_to_image

# 模型生成（[-1, 1] 範圍）
generated = model.sample(...)  # Shape: (1, 1, 128, 256), range [-1, 1]

# 自動處理範圍轉換
img = piano_roll_to_image(
    generated.squeeze(),  # (128, 256)
    apply_colormap=True,
    return_pil=True
)
img.save('output.png')
```

### TensorBoard 記錄
```python
from visualizer import log_piano_roll_to_tensorboard

# 自動處理 [-1, 1] 範圍
log_piano_roll_to_tensorboard(
    writer=tensorboard_writer,
    tag='val/generated',
    piano_roll=generated.squeeze(),  # Can be [-1, 1] or [0, 1]
    global_step=step,
    apply_colormap=True
)
```

### 比較圖片
```python
from visualizer import compare_piano_rolls

# 兩者都可以是 [-1, 1] 或 [0, 1]
compare_piano_rolls(
    original=ground_truth,
    generated=model_output,
    save_path='comparison.png'
)
```

## 影響範圍

### ✅ 已更新
- `visualizer.py` 的所有函數
- 自動檢測和轉換範圍
- 向後兼容 [0, 1] 格式

### ✅ 無需修改
- `inference.py` - 使用 `piano_roll_to_image()`
- `trainer.py` - 使用 `log_piano_roll_to_tensorboard()`
- 所有使用可視化函數的代碼

### ✅ 自動處理
- Training 時的圖片記錄
- Inference 時的圖片儲存
- TensorBoard 可視化
- GIF 生成

## 檢查清單

- [x] `piano_roll_to_image()` 支援 [-1, 1]
- [x] `visualize_piano_roll()` 支援 [-1, 1]
- [x] `compare_piano_rolls()` 支援 [-1, 1]
- [x] `log_piano_roll_to_tensorboard()` 自動處理
- [x] 向後兼容 [0, 1] 格式
- [x] 測試實際模型輸出
- [x] 驗證圖片正確性

## 注意事項

### 範圍檢測邏輯
```python
if piano_roll_np.min() < -0.5:
    # 判定為 [-1, 1] 範圍
```

**假設**：
- [-1, 1] 範圍的數據最小值會 < -0.5
- [0, 1] 範圍的數據最小值會 >= 0

**限制**：
- 如果 [-1, 1] 數據的最小值 >= -0.5，會被誤判為 [0, 1]
- 實際上不太可能發生（需要所有值都 > -0.5）

### 性能影響
- ✅ 檢測和轉換開銷極小
- ✅ 只在轉換為 numpy 後執行一次
- ✅ 不影響訓練或推論速度

## 總結

✅ **所有可視化功能已更新**

- 自動支援 [-1, 1] 範圍
- 向後兼容 [0, 1] 範圍
- 無需修改現有代碼
- 所有測試通過

**現在可以直接使用模型輸出進行可視化，無需手動轉換範圍！**

---

**更新版本**: 1.0  
**更新日期**: 2025-11-07  
**狀態**: ✅ 完成

