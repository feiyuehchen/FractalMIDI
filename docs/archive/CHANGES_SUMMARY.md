# FractalMIDI 修正摘要

## 修正日期
2025-01-18

## 問題診斷

經過深入分析，發現以下關鍵問題：

### 1. 推理初始化策略錯誤
**問題：** 推理時使用 `-1` 初始化 canvas，但 `-1` 是訓練時的 mask token，不應該用於 unconditional generation。

**影響：** 模型可能認為所有位置都需要"填充"，導致生成過於密集的音符。

**修正：** 將推理初始化改為 `0`（靜音），讓模型從空白畫布開始生成。

### 2. velocity_loss 輸出範圍錯誤
**問題：** velocity_loss 將 [0, 255] 映射到 [-1, 1]，但訓練資料使用 [0, 1] 範圍。

**影響：** 輸出範圍與訓練不一致，導致生成結果異常。

**修正：** 將輸出範圍改為 [0, 1]，與訓練保持一致。

### 3. 硬編碼寬度限制
**問題：** 多處硬編碼 256 寬度，無法支援 128×128 或 128×512 等其他尺寸。

**影響：** 只能生成固定尺寸，無法靈活使用。

**修正：** 新增 `target_width` 參數，支援任意寬度 (128, 256, 512, etc.)。

## 修改的檔案

### 核心模型檔案

#### 1. `models/velocity_loss.py`
**修改位置：** Line 115-171

**變更內容：**
- 初始化從 `-1.0` 改為 `0.0`（靜音）
- 輸出範圍從 `[-1, 1]` 改為 `[0, 1]`
- 移除不必要的範圍轉換

```python
# 修改前
velocity_values = torch.full((bsz, 1), -1.0, ...)
velocity_values[:, 0] = (sampled_ids.float() / 255.0) * 2.0 - 1.0

# 修改後
velocity_values = torch.zeros((bsz, 1), ...)
velocity_values[:, 0] = sampled_ids.float() / 255.0
```

#### 2. `models/ar_generator.py`
**修改位置：** Line 221-322

**變更內容：**
- 新增 `target_width` 參數支援
- 初始化從 `-1.0` 改為 `0.0`
- 使用動態計算的 `actual_seq_len` 取代硬編碼 `self.seq_len`

```python
# 修改前
canvas = torch.full((bsz, self.seq_len, ...), -1.0, ...)

# 修改後
h_patches = self.img_size // self.patch_size
w_patches = target_width // self.patch_size
actual_seq_len = h_patches * w_patches
canvas = torch.zeros((bsz, actual_seq_len, ...), ...)
```

#### 3. `models/mar_generator.py`
**修改位置：** Line 273-563

**變更內容：**
- 新增 `target_width` 參數支援
- 初始化從 `-1.0` 改為 `0.0`
- 移除硬編碼的 `w_patches = h_patches * 2` 假設
- 動態計算 `w_patches` 和 `actual_seq_len`

```python
# 修改前
w_patches = h_patches * 2 if _current_level == 0 else h_patches
patches = torch.full(..., -1.0, ...)

# 修改後
w_patches = target_width // self.patch_size
patches = torch.zeros(..., ...)
```

#### 4. `models/fractal_gen.py`
**修改位置：** Line 148-223

**變更內容：**
- 新增 `target_width` 參數（預設 256）
- Canvas 初始化使用動態寬度
- 將 `target_width` 參數傳遞到所有子層級

```python
# 修改前
'canvas': torch.full((batch_size, 1, 128, 256), -1.0)

# 修改後
'canvas': torch.zeros((batch_size, 1, 128, target_width))
```

### 推理腳本

#### 5. `inference.py`
**修改位置：** Line 174-175, 272

**變更內容：**
- 新增 `--target_width` 命令列參數
- 傳遞 `target_width` 到 `model.sample()` 調用

```python
parser.add_argument('--target_width', type=int, default=256,
                   help='Target width for generation (128, 256, 512, etc.)')

generated = model.model.sample(..., target_width=args.target_width)
```

#### 6. `run_inference.sh`
**修改位置：** Line 28, 47, 64

**變更內容：**
- 新增 `TARGET_WIDTH` 環境變數
- 顯示在配置輸出中
- 傳遞給 inference.py

```bash
TARGET_WIDTH=256  # Options: 128, 256, 512
--target_width "$TARGET_WIDTH"
```

## 新增的檔案

### `TRAINING_VERSIONS.md`
完整文檔說明：
- 128×128 和 256×256 兩種訓練版本
- 數值語義 (-1=mask, 0=靜音, (0,1]=velocity)
- 推理配置範例
- 架構層級說明
- Generator 類型和掃描順序
- 記憶體考量和最佳實踐

## 數值範圍語義釐清

根據用戶確認，正確的語義是：

| 數值 | 訓練時 | 推理時 | 意義 |
|------|--------|--------|------|
| `-1` | ✓ Mask token | ✗ 不使用 | 需要預測的位置（白色） |
| `0` | ✓ 靜音 | ✓ 初始值 | 沒有音符（黑色） |
| `(0, 1]` | ✓ Velocity | ✓ 輸出 | 音符力度 |

**關鍵洞察：** `-1` 只在訓練時作為 mask token 使用，推理時應該從 `0` 開始。

## 測試建議

### 1. 快速驗證測試

```bash
# 測試 128×128 生成（使用已訓練的 128×128 模型）
python inference.py \
    --checkpoint path/to/model.ckpt \
    --mode unconditional \
    --num_samples 1 \
    --target_width 128 \
    --generation_length 128 \
    --output_dir outputs/test_128x128

# 測試 128×256 生成（使用已訓練的 256 模型）
python inference.py \
    --checkpoint path/to/model.ckpt \
    --mode unconditional \
    --num_samples 1 \
    --target_width 256 \
    --generation_length 256 \
    --output_dir outputs/test_128x256
```

### 2. 分析檢查項目

執行生成後，檢查：

```python
import symusic

score = symusic.Score('outputs/test_128x128/unconditional_000.mid')
track = score.tracks[0]

# 檢查音符密度
num_notes = len(track.notes)
duration_16ths = 128  # 或 256
notes_per_step = num_notes / duration_16ths

print(f"Total notes: {num_notes}")
print(f"Notes per step: {notes_per_step:.2f}")
print(f"Should be: 1-5 notes per step (reasonable range)")

# 檢查音高範圍
pitches = [note.pitch for note in track.notes]
print(f"Pitch range: [{min(pitches)}, {max(pitches)}]")
print(f"Should be: [21-108] for typical piano music")

# 檢查 velocity 分佈
velocities = [note.velocity for note in track.notes]
print(f"Velocity range: [{min(velocities)}, {max(velocities)}]")
print(f"Should be: [20-120] with variation")
```

### 3. 預期改善

修正後應該看到：

**音符密度：**
- 修正前：~20 notes/step（過於密集）
- 修正後：~1-5 notes/step（合理範圍）

**音高範圍：**
- 修正前：0-127（不合理的全範圍）
- 修正後：21-108（鋼琴常用範圍）

**音符持續時間：**
- 修正前：過長或不規則
- 修正後：符合音樂節奏（120-480 ticks）

**整體品質：**
- 更清晰的和聲結構
- 更自然的節奏模式
- 更好的靜音分佈

## 額外優化建議

如果生成品質仍需改善，可以調整以下參數：

### 1. 增加迭代次數
```bash
--num_iter_list 20 12 8 2  # 從 12 8 4 1 增加
```

### 2. 調整溫度
```bash
--temperature 0.8  # 從 1.0 降低，產生更確定性的輸出
```

### 3. 使用 sparsity bias（如果仍然太密集）
```bash
--sparsity_bias 0.1  # 輕微鼓勵稀疏性
```

### 4. 嘗試不同的 scan_order
對於 AR 模型，可以比較：
- `row_major`: 音高優先（適合旋律）
- `column_major`: 時間優先（適合和弦）

## 向後兼容性

這些修改主要影響推理行為，不影響：
- 已訓練的模型權重（無需重新訓練）
- 訓練腳本和流程
- Dataset 處理邏輯

但需要注意：
- 舊的推理腳本需要更新以包含 `target_width` 參數
- 如果沒有提供 `target_width`，預設使用 256

## 未來工作

可能的進一步改善：

1. **Adaptive mask ratio**：根據音樂複雜度動態調整
2. **Multi-scale training**：同時訓練多種寬度
3. **Better sparsity control**：在 logits 層面直接控制音符密度
4. **Conditional generation**：支援風格、節奏、和聲等條件
5. **Real-time generation**：優化推理速度支援互動式生成

## 參考

- Original issue: 生成音符過於密集，音高範圍異常
- Root cause: 推理初始化策略和輸出範圍錯誤
- Solution: 統一數值語義，支援可變寬度
- Status: ✓ 修正完成，待測試驗證

