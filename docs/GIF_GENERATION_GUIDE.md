# GIF 生成過程可視化指南

## 功能說明

現在可以記錄模型生成過程的每個 iteration，並製作成 GIF 動畫，用於觀察模型如何逐步生成音樂。

## 使用方法

### 1. 使用腳本

最簡單的方式是使用 `run_inference.sh`：

```bash
# 編輯 run_inference.sh，確保這行沒有被註解：
SAVE_GIF="--save_gif"

# 運行推理
bash run_inference.sh
```

### 2. 直接使用 Python

```bash
python inference.py \
    --checkpoint path/to/model.ckpt \
    --mode unconditional \
    --num_samples 5 \
    --save_gif \
    --target_width 256 \
    --output_dir outputs/with_gif
```

**注意：** `--save_gif` 只會為**第一個樣本**生成 GIF，以節省時間和空間。

## 輸出文件

生成完成後，在輸出目錄會看到：

```
outputs/with_gif/
├── unconditional_000.mid           # MIDI 文件
├── unconditional_000.png           # 最終生成結果圖片
├── unconditional_000_process.gif   # 生成過程 GIF ⭐
├── unconditional_001.mid
├── unconditional_001.png
├── unconditional_002.mid
├── ...
```

**只有第一個樣本** (`unconditional_000`) 會有 `_process.gif` 文件。

## GIF 內容說明

GIF 中每一幀顯示：
- **圖像**：當前生成的 piano roll 狀態
- **文字標註**：當前所在的層級和 iteration
  - 例如："AR L0 patch 45/256" 表示 Level 0 的第 45 個 patch
  - 或 "Iter 3 batch 2" 表示第 3 次 iteration 的第 2 個 batch

### 生成過程階層

對於 4 層架構 (128→16→4→1)：

1. **Level 0** (128×256 → patches):
   - 生成 16×16 的 patches
   - 按照 scan_order 順序（row_major 或 column_major）
   
2. **Level 1-3**: 
   - 遞歸生成更小的 patches
   - 最終到 1×1 的 velocity 值

## GIF 參數調整

### 修改 FPS（每秒幀數）

編輯 `inference.py` 中的 `create_generation_gif` 調用：

```python
create_generation_gif(intermediates, gif_path, fps=2)  # 預設 2 FPS
```

可以改為：
- `fps=1`: 更慢，更容易觀察細節
- `fps=5`: 更快，適合快速瀏覽
- `fps=10`: 很快

### 減少幀數

如果 GIF 太大或幀數太多，可以在生成器代碼中調整記錄頻率：

**MAR Generator** (`models/mar_generator.py`):
```python
# Line 378: 調整 batch_generate_size
batch_generate_size = 2  # 改為 4 或更大，減少幀數
```

**AR Generator** (`models/ar_generator.py`):
```python
# Line 258: 調整 record_interval
record_interval = max(1, actual_seq_len // 8)  # 改為 // 4，記錄更多幀
                                                # 改為 // 16，記錄更少幀
```

## 使用場景

### 1. 調試模型行為

觀察 GIF 可以發現：
- 模型是否按預期順序生成
- 哪些區域先被填充
- 生成過程是否平滑
- 是否有突然的變化或錯誤

### 2. 比較不同配置

生成多個 GIF 比較：

```bash
# MAR with row_major
python inference.py --checkpoint model.ckpt --save_gif \
    --output_dir outputs/mar_row

# AR with column_major  
python inference.py --checkpoint model.ckpt --save_gif \
    --output_dir outputs/ar_col
```

然後比較兩個 `_process.gif` 的生成模式差異。

### 3. 展示用途

GIF 可以用於：
- 論文/報告中展示生成過程
- 演講時的可視化演示
- 幫助他人理解模型工作原理

## 範例工作流程

```bash
# 1. 生成帶 GIF 的樣本
python inference.py \
    --checkpoint outputs/my_model/checkpoints/best.ckpt \
    --mode unconditional \
    --num_samples 10 \
    --save_gif \
    --num_iter_list 12 8 4 1 \
    --temperature 0.9 \
    --target_width 256 \
    --output_dir outputs/test_with_gif

# 2. 查看 GIF
# 使用瀏覽器或圖片查看器打開：
# outputs/test_with_gif/unconditional_000_process.gif

# 3. 如果生成品質好，生成更多樣本（不需要 GIF）
python inference.py \
    --checkpoint outputs/my_model/checkpoints/best.ckpt \
    --mode unconditional \
    --num_samples 100 \
    --temperature 0.9 \
    --target_width 256 \
    --output_dir outputs/final_samples
```

## 技術細節

### 幀記錄機制

- **Level 0**: 每個 iteration 或每個 patch 記錄一幀
- **其他 Levels**: 僅在更新到 Level 0 canvas 時記錄
- **Canvas**: 使用累積的 canvas，顯示當前完整狀態

### 記憶體考量

記錄中間幀會增加記憶體使用：
- 每幀約 128×256×4 bytes (對於 128×256 piano roll)
- 50 幀 ≈ 6.5 MB
- 通常不會造成問題

如果記憶體不足：
- 只生成少量樣本 (`--num_samples 1`)
- 減少 `num_iter_list` 的值
- 使用較小的 `target_width`

### 幀順序說明

GIF 中的幀按照**生成順序**排列：

**MAR (Masked Autoregressive)**:
1. Init (全黑或全白)
2. Iteration 1 - batch 1
3. Iteration 1 - batch 2
4. ...
5. Iteration 2 - batch 1
6. ...

**AR (Autoregressive)**:
1. Init
2. Patch 1
3. Patch 2
4. ...
5. Patch N

## 進階用法

### 自定義 GIF 外觀

修改 `inference.py` 中的 `create_generation_gif` 函數：

```python
def create_generation_gif(intermediates, output_path, fps=2):
    # 添加更多視覺元素
    # 例如：進度條、時間戳、統計資訊等
    
    # 修改文字樣式
    font = ImageFont.truetype("path/to/font.ttf", 20)  # 更大字體
    
    # 添加背景色
    draw.rectangle(..., fill='blue')  # 改變背景色
```

### 保存中間幀為圖片

如果想要保存每一幀為單獨的圖片：

```python
# 在 create_generation_gif 中添加
for idx, frame_data in enumerate(intermediates):
    img = ...  # 創建圖片
    img.save(f"frames/frame_{idx:04d}.png")
```

## 常見問題

### Q: GIF 文件太大怎麼辦？

**A:** 
- 減少 FPS: `fps=1`
- 減少 iterations: `--num_iter_list 8 4 2 1`
- 減少記錄頻率（修改 batch_generate_size）
- 壓縮 GIF（使用 gifsicle 等工具）

### Q: GIF 播放太快/太慢？

**A:** 修改 `fps` 參數在 `inference.py` 中：
```python
create_generation_gif(intermediates, gif_path, fps=1)  # 慢
create_generation_gif(intermediates, gif_path, fps=5)  # 快
```

### Q: 可以為所有樣本生成 GIF 嗎？

**A:** 可以，但會很慢。修改 `inference.py`:
```python
# 改為對所有樣本啟用
return_intermediates = args.save_gif  # 移除 (i == 0 and ...)
```

### Q: GIF 沒有生成？

**A:** 檢查：
1. 是否使用了 `--save_gif` 標誌
2. 檢查終端輸出是否有錯誤
3. 確保 `imageio` 已安裝: `pip install imageio`
4. 檢查輸出目錄權限

### Q: 想看 Level 1-3 的細節怎麼辦？

**A:** 當前實現只記錄 Level 0 的累積結果。如果需要看其他層級，需要修改代碼在 `mar_generator.py` 和 `ar_generator.py` 中添加更多記錄點。

## 依賴套件

確保安裝了以下套件：

```bash
pip install imageio
pip install Pillow  # 通常已安裝
```

## 總結

GIF 生成功能讓您可以：
- ✅ 觀察模型生成過程
- ✅ 調試和理解模型行為
- ✅ 比較不同配置的效果
- ✅ 用於展示和教學

只需添加 `--save_gif` 標誌即可！🎬

