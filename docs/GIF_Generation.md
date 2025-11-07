# Generation GIF 動畫功能

## 功能概述

在訓練過程中，每當到達 `log_images_every_n_steps` 時，模型會在 validation 階段生成樣本，並記錄整個生成過程的中間步驟，最後製作成 GIF 動畫。

## 實作細節

### 1. 啟用 intermediate 記錄

在 `trainer.py` 的 `_log_images` 方法中：

```python
generated, intermediates = self.model.sample(
    batch_size=samples_to_generate,
    cond_list=None,
    num_iter_list=[8, 4, 2, 1],
    cfg=1.0,
    cfg_schedule="constant",
    temperature=1.0,
    filter_threshold=0,
    return_intermediates=True  # ← 啟用中間步驟記錄
)
```

### 2. 生成 GIF

`_create_generation_gifs` 方法處理：
1. 從 intermediates 中提取每個樣本的所有中間步驟
2. 將每個步驟正規化到 [0, 255]
3. 應用 viridis colormap
4. 使用 PIL 製作 GIF 動畫
5. 儲存到磁碟並記錄預覽到 TensorBoard

### 3. GIF 參數

```python
frames[0].save(
    buf,
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=200,  # 200ms 每幀
    loop=0         # 無限循環
)
```

## 生成的檔案

### 目錄結構

```
outputs/fractalgen_ar_ar_ar_ar/
└── lightning_logs/
    └── version_0/
        ├── checkpoints/
        │   └── step_00010000-val_loss_0.xxxx.ckpt
        ├── generation_gifs/  ← GIF 儲存在這裡
        │   ├── step_0010000_sample_0.gif
        │   ├── step_0010000_sample_1.gif
        │   ├── step_0010000_sample_2.gif
        │   └── step_0010000_sample_3.gif
        └── events.out.tfevents...
```

### 檔案命名

- `step_{step:07d}_sample_{idx}.gif`
- 例如：`step_0010000_sample_0.gif` 表示第 10000 步的第 0 個樣本

## TensorBoard 可視化

### 1. 最終生成結果
```
val/generated/step_0010000_sample_0
val/generated/step_0010000_sample_1
...
```

### 2. GIF 預覽
```
val/generation_preview/step_0010000_sample_0
val/generation_preview/step_0010000_sample_1
...
```
顯示 GIF 的最後一幀作為靜態預覽。

### 3. Ground Truth
```
val/ground_truth/step_0010000_sample_0
val/ground_truth/step_0010000_sample_1
...
```

## 配置選項

### log_images_every_n_steps

控制多久生成一次圖片和 GIF：

```bash
python main.py \
    --log_images_every_n_steps 5000 \  # 每 5000 步生成一次
    --num_images_to_log 4               # 每次生成 4 個樣本
```

### 與 validation 的關係

- GIF 只在 **validation 時** 生成
- 必須符合兩個條件：
  1. `global_step % log_images_every_n_steps == 0`
  2. 當前是 validation step

例如：
- `val_check_interval_steps = 2000`
- `log_images_every_n_steps = 5000`
- 實際生成 GIF 的步數：10000, 20000, 30000, ...

## GIF 內容解析

### 階層式生成過程

每個 GIF 展示了 FractalGen 的階層式生成：

1. **第 1-8 幀**：Level 0 (128x128) 的迭代生成
   - 從粗略的結構開始
   - 逐步填充細節

2. **第 9-12 幀**：Level 1 (16x16) 的細化
   - 在更高解析度上生成細節

3. **第 13-14 幀**：Level 2 (4x4) 的進一步細化

4. **最後一幀**：Level 3 (1x1) velocity 預測

### 視覺效果

- **Colormap**: viridis (藍 → 綠 → 黃)
  - 藍色：沒有音符 (velocity = 0)
  - 黃色：強音符 (velocity = 1)
  
- **座標軸**：
  - 水平軸：時間 (time steps)
  - 垂直軸：音高 (pitch, 0-127)

## 除錯用途

### 1. 檢查生成過程

透過 GIF 可以看到：
- 模型是否從有意義的粗略結構開始
- 細節是否逐步增加
- 是否有突然的變化或異常

### 2. 訓練進度追蹤

比較不同 step 的 GIF：
```bash
step_0005000_sample_0.gif  # 早期：可能較粗糙
step_0010000_sample_0.gif  # 中期：開始有結構
step_0050000_sample_0.gif  # 後期：更精細、更連貫
```

### 3. 架構驗證

- **AR 模型**：應該看到序列生成，從左到右逐步填充
- **MAR 模型**：應該看到並行生成，所有位置同時出現

## 性能考量

### 記憶體使用

生成 GIF 會增加記憶體使用：
- 需要保存所有中間步驟
- 4 個樣本 × 20 幀 × (128×256) ≈ 26MB

建議：
- `num_images_to_log` 不要設太大 (建議 ≤ 4)
- `log_images_every_n_steps` 設大一點 (≥ 5000)

### 生成時間

製作 GIF 需要額外時間：
- 每個樣本約 1-2 秒
- 4 個樣本約 5-10 秒

不會阻塞訓練，因為在 validation 時執行。

## 範例輸出

### 正常的生成 GIF

```
Frame 0:  [粗略的結構，可能只有幾個大塊]
Frame 4:  [開始出現更多細節]
Frame 8:  [Level 0 完成，有基本的音符結構]
Frame 12: [Level 1 完成，音符更清晰]
Frame 16: [接近最終結果]
Frame 20: [最終結果，velocity 已確定]
```

### 異常的生成 GIF

如果看到以下情況，可能有問題：
- 所有幀都是全黑或全綠
- 突然出現劇烈變化
- 沒有從粗到細的過程

## 查看 GIF

### 在終端中

```bash
# 使用 feh (Linux)
feh outputs/.../generation_gifs/step_0010000_sample_0.gif

# 使用 Firefox
firefox outputs/.../generation_gifs/step_0010000_sample_0.gif
```

### 在 Python 中

```python
from PIL import Image

gif = Image.open('step_0010000_sample_0.gif')
gif.show()
```

### 轉換為影片

如果需要更好的播放體驗：

```bash
ffmpeg -i step_0010000_sample_0.gif \
       -vf "fps=5,scale=512:1024:flags=neighbor" \
       -c:v libx264 -pix_fmt yuv420p \
       step_0010000_sample_0.mp4
```

## 總結

GIF 生成功能提供了：
- ✅ 視覺化階層式生成過程
- ✅ 除錯和分析工具
- ✅ 訓練進度的直觀展示
- ✅ 不影響訓練速度（只在 validation 時執行）

這對於理解和改進 FractalGen 模型非常有幫助！


