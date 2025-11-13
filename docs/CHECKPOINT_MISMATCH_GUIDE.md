# Checkpoint 不匹配問題診斷與修復指南

## 問題症狀

生成的 MIDI 出現以下異常：
- ❌ 音符過於密集（每個時間步 20-30 個音符）
- ❌ 音高範圍異常（使用 0-127 全範圍）
- ❌ Velocity 偏高或偏低
- ❌ 音符時長不自然

## 根本原因

### Checkpoint 與代碼不匹配

您的 checkpoint 是用**修正前的代碼**訓練的：

**訓練時（舊代碼）：**
```python
# 推理可能使用 -1 初始化
canvas = torch.full(..., -1.0, ...)
# velocity_loss 可能輸出 [-1, 1]
velocity = (sampled_ids / 255.0) * 2.0 - 1.0
```

**推理時（新代碼）：**
```python
# 現在使用 0 初始化
canvas = torch.zeros(...)
# velocity_loss 現在輸出 [0, 1]
velocity = sampled_ids / 255.0
```

**結果：** 模型在推理時看到與訓練時不同的數值分佈，導致輸出異常。

## 解決方案

### 方案 A：調整推理參數（臨時方案）⚡

**適用場景：**
- 想快速測試現有 checkpoint
- 暫時還不想重新訓練
- 評估修復效果

**步驟：**

```bash
# 使用調整後的參數
bash run_inference_tuned.sh
```

**調整的參數：**
1. **Temperature**: 1.0 → 0.7（降低隨機性）
2. **Sparsity bias**: 0.0 → 0.5（減少音符密度）
3. **Velocity threshold**: 0.1 → 0.2（過濾弱音符）
4. **Num iterations**: 12,8,4,1 → 20,12,8,2（提高品質）

**優點：**
- ✅ 快速測試
- ✅ 不需要重新訓練
- ✅ 可以評估代碼修改效果

**缺點：**
- ❌ 治標不治本
- ❌ 可能仍有問題
- ❌ 參數調整困難

### 方案 B：重新訓練（推薦方案）⭐

**適用場景：**
- 需要最佳生成品質
- 有足夠的計算資源
- 想充分利用代碼修正

**步驟：**

#### 1. 選擇訓練尺寸

```bash
# 小模型（快速測試）
bash train_128x128.sh

# 大模型（更好品質）
bash train_256x256.sh
```

#### 2. 監控訓練

```bash
# 查看 TensorBoard
tensorboard --logdir outputs/fractalgen_128x128/logs
```

觀察：
- Loss 曲線下降
- 生成樣本品質
- 各層的統計資訊

#### 3. 測試新 checkpoint

```bash
python inference.py \
    --checkpoint outputs/fractalgen_128x128/checkpoints/step_00050000-val_loss_0.0123.ckpt \
    --num_samples 10 \
    --save_gif \
    --output_dir outputs/new_model_test
```

**優點：**
- ✅ 根本解決問題
- ✅ 充分利用修正後的代碼
- ✅ 最佳生成品質

**缺點：**
- ❌ 需要時間（2-7 天）
- ❌ 需要計算資源

### 方案 C：使用舊版推理代碼（不推薦）❌

**理由：** 
- 修正是為了解決問題，不應該回退
- 舊代碼有已知的 bug
- 長期來看不是好的解決方案

## 診斷工具

### 快速檢查生成品質

創建診斷腳本：

```python
# diagnose_midi.py
import symusic
import numpy as np
import sys

def diagnose_midi(midi_path):
    score = symusic.Score(midi_path)
    
    print(f"\n{'='*60}")
    print(f"Diagnosing: {midi_path}")
    print(f"{'='*60}")
    
    for track in score.tracks:
        if len(track.notes) == 0:
            continue
            
        notes = track.notes
        pitches = [n.pitch for n in notes]
        velocities = [n.velocity for n in notes]
        
        # Calculate statistics
        notes_per_step = len(notes) / 256  # Assuming 256 steps
        
        print(f"\n📊 Statistics:")
        print(f"  Total notes: {len(notes)}")
        print(f"  Notes/step: {notes_per_step:.2f}")
        print(f"  Pitch range: [{min(pitches)}, {max(pitches)}]")
        print(f"  Velocity mean: {np.mean(velocities):.1f}")
        
        print(f"\n✅ Quality Check:")
        
        # Check note density
        if 1 <= notes_per_step <= 8:
            print(f"  ✓ Note density OK ({notes_per_step:.1f} notes/step)")
        elif notes_per_step < 1:
            print(f"  ⚠ Too sparse ({notes_per_step:.1f} notes/step)")
        else:
            print(f"  ✗ Too dense ({notes_per_step:.1f} notes/step)")
        
        # Check pitch range
        if 20 <= min(pitches) <= 40 and 70 <= max(pitches) <= 110:
            print(f"  ✓ Pitch range OK")
        else:
            print(f"  ⚠ Unusual pitch range: [{min(pitches)}, {max(pitches)}]")
        
        # Check velocity
        if 30 <= np.mean(velocities) <= 90:
            print(f"  ✓ Velocity OK")
        else:
            print(f"  ⚠ Unusual velocity: {np.mean(velocities):.1f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_midi.py path/to/file.mid")
        sys.exit(1)
    
    diagnose_midi(sys.argv[1])
```

**使用：**
```bash
python diagnose_midi.py outputs/inference_arararar_row_major/unconditional_000.mid
python diagnose_midi.py outputs/inference_tuned/unconditional_000.mid
```

## 比較不同方案的結果

### 方案 A 效果預期

調整參數後：
- 音符密度：32 → 8-15 notes/step（改善但仍偏高）
- 音高範圍：可能仍然較寬
- 整體品質：中等

### 方案 B 效果預期

重新訓練後：
- 音符密度：1-5 notes/step（理想）
- 音高範圍：21-108（合理）
- 整體品質：最佳

## 建議的工作流程

### 短期（1-2 小時）

```bash
# 1. 測試調整參數的效果
bash run_inference_tuned.sh

# 2. 診斷結果
python diagnose_midi.py outputs/inference_tuned/unconditional_000.mid

# 3. 評估是否可接受
# 播放 MIDI 聽聽看
```

### 中期（1 週）

```bash
# 1. 開始訓練新模型（先小模型）
bash train_128x128.sh

# 2. 定期檢查訓練進度
tensorboard --logdir outputs/fractalgen_128x128/logs

# 3. 在 50k steps 左右測試中間結果
python inference.py \
    --checkpoint outputs/fractalgen_128x128/checkpoints/step_00050000-*.ckpt \
    --num_samples 5 \
    --output_dir outputs/test_50k

# 4. 診斷並決定是否繼續訓練
python diagnose_midi.py outputs/test_50k/unconditional_000.mid
```

### 長期（2-4 週）

```bash
# 1. 訓練大模型（如果需要）
bash train_256x256.sh

# 2. 比較不同配置
# - MAR vs AR
# - row_major vs column_major
# - 不同的 mask_ratio

# 3. 選擇最佳模型用於生產
```

## 常見問題

### Q: 調整參數後仍然很密集怎麼辦？

**A:** 嘗試：
1. 進一步提高 sparsity_bias: 0.5 → 0.7
2. 降低 temperature: 0.7 → 0.5
3. 提高 velocity_threshold: 0.2 → 0.3
4. 考慮重新訓練

### Q: 必須重新訓練嗎？

**A:** 如果：
- 參數調整無法達到滿意效果
- 需要最佳品質
- 有足夠的計算資源

則建議重新訓練。

### Q: 訓練需要多久？

**A:** 
- 128×128: 2-4 天（雙 GPU）
- 256×256: 4-7 天（雙 GPU）
- 可以先訓練 50k steps 看效果

### Q: 有沒有其他臨時方案？

**A:** 可以嘗試：
1. 後處理：過濾過密的音符
2. 量化：將 velocity 限制在合理範圍
3. 音高過濾：只保留 21-108 範圍

但這些都是治標不治本。

## 代碼修改總結

我們做的關鍵修改：

### 1. 初始化策略
```python
# 前：-1 (mask token)
# 後：0 (silence)
```

### 2. velocity_loss 輸出
```python
# 前：[-1, 1]
# 後：[0, 1]
```

### 3. 支援可變寬度
```python
# 前：硬編碼 256
# 後：target_width 參數
```

這些修改**需要重新訓練**才能完全發揮效果。

## 結論

**短期建議：**
- 嘗試 `run_inference_tuned.sh`
- 評估結果是否可接受

**長期建議：**
- 重新訓練模型
- 使用修正後的代碼
- 獲得最佳生成品質

**當前 checkpoint 的狀態：**
- ⚠️ 與新代碼不匹配
- ⚠️ 生成結果異常
- ✅ 可通過參數部分緩解
- ❌ 最終需要重新訓練

