# FractalGen MIDI 🎹

基於 FractalGen 架構的 MIDI 音樂生成模型。使用階層式生成方法，從粗到細逐步生成 piano roll。

## 📁 文件結構

```
FractalMIDI/
├── model.py                    # FractalGen 模型實現（3層遞歸架構）
├── trainer.py                  # PyTorch Lightning 訓練器
├── dataset.py                  # MIDI 數據加載與預處理
├── visualizer.py               # Piano roll 可視化工具
├── main_fractalgen.py          # 訓練主程序（推薦使用）
├── inference_fractalgen.py     # 推理程序（推薦使用）
├── main.py                     # 原訓練程序（保留）
└── inference.py                # 原推理程序（保留）
```

## ⚡ 快速開始

### 1. 環境設置

```bash
# 創建環境
conda create -n frac python=3.10
conda activate frac

# 安裝依賴
pip install -r requirements.txt
```

### 2. 準備數據

```bash
# 創建數據集列表
find /path/to/midi/files -name "*.mid" > dataset/train.txt
find /path/to/validation/files -name "*.mid" > dataset/valid.txt
```

### 3. 開始訓練

```bash
# Small model (30M 參數，推薦用於快速實驗)
python main_fractalgen.py \
    --model_size small \
    --batch_size 8 \
    --max_epochs 50 \
    --devices 0,1

# Base model (56M 參數，推薦用於正式訓練)
python main_fractalgen.py \
    --model_size base \
    --batch_size 4 \
    --max_epochs 100 \
    --devices 0,1

# Large model (90M 參數，最佳質量)
python main_fractalgen.py \
    --model_size large \
    --batch_size 2 \
    --max_epochs 100 \
    --devices 0,1
```

### 4. 監控訓練

```bash
tensorboard --logdir outputs/fractalgen
```

### 5. 生成 MIDI

```bash
# 無條件生成
python inference_fractalgen.py \
    --checkpoint outputs/fractalgen/checkpoints/last.ckpt \
    --mode unconditional \
    --num_samples 10 \
    --generation_length 256 \
    --save_images

# 有條件生成（基於前綴）
python inference_fractalgen.py \
    --checkpoint outputs/fractalgen/checkpoints/last.ckpt \
    --mode conditional \
    --condition_midi input.mid \
    --condition_length 64 \
    --generation_length 256 \
    --save_images

# Inpainting（局部重新生成）
python inference_fractalgen.py \
    --checkpoint outputs/fractalgen/checkpoints/last.ckpt \
    --mode inpainting \
    --input_midi input.mid \
    --mask_start 64 \
    --mask_end 192 \
    --save_images
```

## 🏗️ 模型架構

### FractalGen 階層結構

```
Level 0 (128 patches): PianoRollMAR
    ↓ (每個 patch 生成下一層的條件)
Level 1 (4 patches): PianoRollMAR
    ↓ (每個 patch 生成像素值)
Level 2 (1 patch): PianoRollVelocityLoss
```

### 關鍵特性

- **階層式生成**: 粗到細，逐步精緻化
- **MAR (Masked Autoregressive)**: 使用遮罩策略進行訓練
- **Iterative Refinement**: 生成時逐步填充被遮罩的區域
- **Classifier-Free Guidance**: 提升生成質量
- **可變長度支持**: 自動處理不同長度的輸入

### 模型規模

| Model | Parameters | Embed Dims | Blocks | Heads | Memory | Speed |
|-------|-----------|------------|--------|-------|--------|-------|
| Small | 30M | 768/384/192 | 16/4/2 | 12/6/3 | ~8GB | 最快 |
| Base | 56M | 1024/512/256 | 24/6/3 | 16/8/4 | ~12GB | 中等 |
| Large | 90M | 1280/640/320 | 32/8/4 | 20/10/5 | ~18GB | 較慢 |

## 🎛️ 重要參數

### 訓練參數

```python
--model_size small/base/large  # 模型規模
--batch_size 8                 # 批次大小（根據 GPU 記憶體調整）
--max_epochs 50                # 訓練輪數
--lr 1e-4                      # 學習率
--weight_decay 0.05            # 權重衰減
--warmup_epochs 5              # Warmup 輪數
--grad_clip 3.0                # 梯度裁剪
--devices 0,1                  # 使用的 GPU
--precision 32                 # 精度 (32/16/bf16)
```

### 生成參數

```python
--num_iter_list 12 8 1         # 每層的迭代次數 [Level0, Level1, Level2]
--cfg 1.0                      # Classifier-free guidance 強度
--temperature 1.0              # 採樣溫度
--sparsity_bias 2.0            # 稀疏性偏置（越高越稀疏）
```

## 📊 數據格式

### Piano Roll 格式

- **形狀**: `(1, 128, T)`
  - 通道: 1 (velocity)
  - 高度: 128 (MIDI 音高 0-127)
  - 寬度: T (時間步，單位為 16th note)
- **值域**: `[0, 1]`（歸一化的 velocity）

### Patch 劃分

使用 `patch_size=4` 將 piano roll 劃分為 patches:
- 128x256 → 2048 patches (32x16)
- 128x128 → 1024 patches (32x8)
- 128x512 → 4096 patches (32x32)

## 🎨 可視化

Piano roll 可視化遵循 Logic Pro 風格：
- **黑色**: 無音符 (velocity = 0)
- **綠色**: 中等力度 (velocity ≈ 64)
- **紅色**: 強力度 (velocity ≈ 127)

## 🔧 配置系統

所有配置使用 dataclass 管理：

```python
# model.py
@dataclass
class MARConfig:
    embed_dim: int = 768
    num_blocks: int = 16
    num_heads: int = 12
    patch_size: int = 4
    # ...

# trainer.py
@dataclass
class FractalTrainerConfig:
    model_size: str = 'small'
    max_epochs: int = 50
    grad_clip: float = 3.0
    # ...

# dataset.py
@dataclass
class DataLoaderConfig:
    batch_size: int = 8
    shuffle: bool = True
    patch_size: int = 4
    # ...
```

## 📈 訓練流程

### 階層式 Loss

```python
# 自動計算所有層的 loss
loss = model(piano_rolls)

# 等價於：
loss = (
    loss_level0 +  # PianoRollMAR (粗層)
    loss_level1 +  # PianoRollMAR (中層)
    loss_level2    # PianoRollVelocityLoss (細層)
)
```

### 採樣流程

```python
# Iterative refinement
for level in [0, 1, 2]:
    for iteration in range(num_iter_list[level]):
        1. 創建遮罩（cosine schedule）
        2. 預測被遮罩的 patches
        3. 填充預測值
        4. 進入下一層或下一次迭代
```

## 🔬 實驗設置

### 推薦設置（Small Model）

```bash
python main_fractalgen.py \
    --model_size small \
    --batch_size 8 \
    --max_epochs 50 \
    --lr 1e-4 \
    --warmup_epochs 5 \
    --devices 0,1 \
    --precision 32
```

### 推薦設置（Base Model）

```bash
python main_fractalgen.py \
    --model_size base \
    --batch_size 4 \
    --max_epochs 100 \
    --lr 8e-5 \
    --warmup_epochs 10 \
    --devices 0,1 \
    --precision 32
```

### 預期 Loss

- **初始**: ~5.5-5.6
- **收斂**: ~1.0-2.0 (取決於數據集)
- **良好**: <0.5

## ⚠️ 已知限制

### 1. 生成維度問題

- **狀態**: 條件生成和 inpainting 的維度匹配需要進一步調試
- **建議**: 先專注於訓練，待模型收斂後再優化生成
- **替代方案**: 使用無條件生成（已可用）

### 2. 長序列

- **限制**: 預設 `max_seq_len=2100`，可處理 128x256 左右的輸入
- **解決**: 如需更長序列，調整 `TrainerConfig.max_seq_len`

### 3. 記憶體使用

- **Small**: ~8GB (batch_size=8)
- **Base**: ~12GB (batch_size=4)
- **Large**: ~18GB (batch_size=2)

## 📚 參考文件

- **詳細技術**: `FRACTALGEN_COMPLETE.md`
- **快速開始**: `QUICK_START_FRACTALGEN.md`
- **當前狀態**: `FRACTAL_STATUS.md`
- **實現總結**: `IMPLEMENTATION_SUMMARY.md`
- **原論文**: https://arxiv.org/abs/2401.05036
- **原代碼**: https://github.com/Yikai-Liao/fractalgen

## 🐛 故障排除

### 訓練 Loss 不下降

```bash
# 檢查數據
python -c "
from dataset import create_dataloader
loader = create_dataloader('dataset/train.txt', batch_size=4)
batch = next(iter(loader))
print(f'Batch shape: {batch[0].shape}')
print(f'Value range: [{batch[0].min():.3f}, {batch[0].max():.3f}]')
"

# 降低學習率
python main_fractalgen.py --lr 5e-5

# 增加 warmup
python main_fractalgen.py --warmup_epochs 10
```

### 記憶體不足

```bash
# 減小批次大小
python main_fractalgen.py --batch_size 2

# 使用小模型
python main_fractalgen.py --model_size small

# 使用混合精度（謹慎）
python main_fractalgen.py --precision bf16
```

### 生成結果太稀疏

```python
# 降低稀疏性偏置
python inference_fractalgen.py --sparsity_bias 1.0

# 調整溫度
python inference_fractalgen.py --temperature 1.2
```

### 生成結果太密集

```python
# 增加稀疏性偏置
python inference_fractalgen.py --sparsity_bias 3.0

# 降低溫度
python inference_fractalgen.py --temperature 0.8
```

## 🎯 最佳實踐

### 1. 數據準備

- 確保 MIDI 文件質量良好
- 建議至少 1000 首曲目
- 驗證集約佔 10-20%

### 2. 訓練策略

- 從 small model 開始驗證流程
- 使用 base model 進行正式訓練
- 監控 TensorBoard 的重建圖像質量

### 3. 生成策略

- 先用無條件生成測試模型
- 調整 `num_iter_list` 平衡速度/質量
- 實驗不同的 `sparsity_bias` 值

### 4. 調參建議

```python
# 快速實驗
num_iter_list = [4, 2, 1]  # 最快

# 平衡質量/速度
num_iter_list = [8, 4, 1]  # 推薦

# 最佳質量
num_iter_list = [16, 8, 1]  # 最慢
```

## 🎓 理解 FractalGen

### 核心概念

1. **階層生成**: 從粗糙到精細，逐層生成
2. **MAR masking**: 訓練時隨機遮罩部分 patches
3. **Iterative refinement**: 生成時逐步填充遮罩區域
4. **Classifier-free guidance**: 混合條件/無條件預測

### 與傳統方法的區別

| 特性 | 傳統 Transformer | FractalGen |
|------|-----------------|------------|
| 生成方式 | 序列化（逐個 token） | 階層化（粗到細） |
| 訓練目標 | 下一個 token | 被遮罩的 patches |
| 採樣 | Autoregressive | Iterative refinement |
| 速度 | O(T) | O(log T) 理論上 |
| 並行性 | 低 | 高 |

## 💡 Tips

### 加速訓練

- 使用多 GPU: `--devices 0,1,2,3`
- 增加 workers: `--num_workers 8`
- 使用 SSD 存儲數據
- 考慮混合精度: `--precision bf16`（實驗性）

### 提升質量

- 增加模型大小: `--model_size large`
- 延長訓練: `--max_epochs 200`
- 數據增強（已內建隨機裁剪）
- 調整採樣參數

### 調試技巧

```bash
# 快速測試（1個 batch）
python main_fractalgen.py --fast_dev_run

# 檢查模型
python -c "
from model import fractalmar_piano_small
model = fractalmar_piano_small()
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
"

# 檢查數據
python -c "
from dataset import create_dataloader
loader = create_dataloader('dataset/train.txt', batch_size=2)
for i, batch in enumerate(loader):
    print(f'Batch {i}: {batch[0].shape}')
    if i >= 2: break
"
```

## 🚀 下一步

### 短期（1-2週）

1. ✅ 開始訓練 small/base model
2. ✅ 監控 loss 和重建圖像
3. ⚠️  調試生成函數維度問題

### 中期（1-2月）

4. 完善 conditional generation
5. 完善 inpainting
6. 優化採樣速度

### 長期（3-6月）

7. 探索更大的模型和數據集
8. 實驗不同的 architecture 變體
9. 發布預訓練模型

---

**🎉 開始你的 FractalGen MIDI 生成之旅吧！**

如有問題，請查閱：
- `FRACTALGEN_COMPLETE.md` - 完整技術細節
- `FRACTAL_STATUS.md` - 當前實現狀態
- `IMPLEMENTATION_SUMMARY.md` - 功能總結

