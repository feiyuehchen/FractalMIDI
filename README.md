# FractalGen MIDI 🎹

基於 FractalGen 架構的 MIDI 音樂生成模型。使用階層式生成方法，從粗到細逐步生成 piano roll。

## 📁 文件結構

```
FractalMIDI/
├── config/                     # 配置檔案目錄
│   ├── train_default.yaml      # 預設訓練配置
│   ├── train_128x128.yaml      # 小尺寸訓練配置
│   ├── train_ar.yaml           # AR 生成器配置
│   └── inference_default.yaml  # 推理配置
├── models/                     # 模組化模型組件
│   ├── attention.py            # 注意力機制
│   ├── blocks.py               # Transformer 區塊
│   ├── mar_generator.py        # MAR 生成器
│   ├── ar_generator.py         # AR 生成器
│   ├── velocity_loss.py        # 力度預測層
│   ├── fractal_gen.py          # 主要 FractalGen 模型
│   ├── generation.py           # 生成函數
│   └── utils.py                # 工具函數
├── docs/                       # 文檔
│   ├── archive/                # 歷史文檔
│   ├── MODEL_STRUCTURE.md      # 模型結構說明
│   ├── TRAINING_GUIDE.md       # 訓練指南
│   └── ...                     # 其他文檔
├── tests/                      # 測試檔案
├── dataset/                    # 數據集列表
├── trainer.py                  # PyTorch Lightning 訓練器
├── dataset.py                  # MIDI 數據加載與預處理
├── visualizer.py               # Piano roll 可視化工具
├── model.py                    # 模型接口（向後兼容）
├── main.py                     # 訓練主程序
├── inference.py                # 推理程序
├── run_training.sh             # 訓練腳本
├── run_inference.sh            # 推理腳本
└── requirements.txt            # 依賴
```

**模組化結構**：模型代碼已重構為模組化結構，提升可讀性和可維護性。詳見 [docs/MODEL_STRUCTURE.md](docs/MODEL_STRUCTURE.md)。

**配置系統**：使用 YAML 配置檔案管理所有超參數，方便實驗管理和版本控制。

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

#### 方法 1: 使用 preprocess.py（推薦）

自動生成訓練/驗證/測試集分割（99.8% / 0.1% / 0.1%）：

```bash
# POP909 資料集
python preprocess.py --dataset pop909

# Aria MIDI 資料集 (aria-midi-v1-unique-ext)
python preprocess.py --dataset ariamidi

# 自訂輸出目錄（可選）
python preprocess.py --dataset ariamidi --output-dir /path/to/output
```

這會自動產生以下檔案：
- `dataset/{dataset_name}/train.txt` - 訓練集檔案列表
- `dataset/{dataset_name}/valid.txt` - 驗證集檔案列表
- `dataset/{dataset_name}/test.txt` - 測試集檔案列表

**支援的資料集：**
- `pop909`: POP909 資料集 (`~/dataset/POP909-Dataset/POP909`)
- `ariamidi`: Aria MIDI v1 資料集 (`~/dataset/aria-midi-v1-unique-ext/data`)

#### 方法 2: 手動創建列表

```bash
# 創建數據集列表
find /path/to/midi/files -name "*.mid" > dataset/train.txt
find /path/to/validation/files -name "*.mid" > dataset/valid.txt
```

### 3. 開始訓練

```bash
# 使用配置檔案（推薦）
bash run_training.sh config/train_default.yaml

# 或使用不同的配置
bash run_training.sh config/train_ar.yaml         # 全 AR 生成器
bash run_training.sh config/train_128x128.yaml    # 小尺寸快速測試

# 直接使用 Python（配置檔案）
python main.py --config config/train_default.yaml

# 覆寫配置中的特定參數
python main.py --config config/train_default.yaml --max_steps 100000 --lr 5e-5

# 或使用命令行參數（向後兼容）
python main.py \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --max_steps 200000 \
    --val_check_interval_steps 2000 \
    --checkpoint_every_n_steps 2000 \
    --devices 0,1 \
    --generator_types "ar,ar,ar,ar" \
    --scan_order "row_major"
```

**可用的配置檔案：**
- `config/train_default.yaml`: 預設配置（128x512, MAR generators）
- `config/train_128x256.yaml`: 中尺寸訓練（128x256, 平衡速度與質量）
- `config/train_128x128.yaml`: 小尺寸訓練（128x128, 更快）

**配置選項：**
- `generator_types`: 每層的生成器類型，可選 `mar` 或 `ar`
- `scan_order`: AR 生成器的掃描順序
  - `row_major`（預設）：先左到右，再上到下（強調時間連續性）
  - `column_major`：先上到下，再左到右（強調和聲結構）

### 4. 監控訓練

```bash
tensorboard --logdir outputs/fractalgen
```

**TensorBoard 中的可視化：**
- `train/loss`, `val_loss`: 訓練和驗證損失
- `val/ground_truth/`: 驗證集的真實 piano rolls
- `val/generated/`: 模型生成的 piano rolls
- `val/generation_preview/`: GIF 動畫的最後一幀預覽

**Generation GIF 動畫：**
在每個 `log_images_every_n_steps` 時，模型會生成帶有中間步驟的動畫 GIF，展示生成過程：
```bash
# GIF 儲存位置
outputs/fractalgen_ar_ar_ar_ar/lightning_logs/version_X/generation_gifs/
├── step_0010000_sample_0.gif
├── step_0010000_sample_1.gif
├── step_0010000_sample_2.gif
└── step_0010000_sample_3.gif
```
每個 GIF 展示模型如何從粗略到精細逐步生成 piano roll，幫助理解階層式生成過程。

### 5. 生成 MIDI

```bash
# 使用腳本（推薦）
bash run_inference.sh outputs/fractalgen/checkpoints/step_00100000.ckpt

# 使用配置檔案
python inference.py \
    --config config/inference_default.yaml \
    --checkpoint outputs/fractalgen/checkpoints/step_00100000.ckpt

# 或使用命令行參數（向後兼容）

# 無條件生成
python inference.py \
    --checkpoint outputs/fractalgen/checkpoints/step_00100000.ckpt \
    --mode unconditional \
    --num_samples 10 \
    --generation_length 256 \
    --save_images

# 有條件生成（基於前綴）
python inference.py \
    --checkpoint outputs/fractalgen/checkpoints/step_00100000.ckpt \
    --mode conditional \
    --condition_midi input.mid \
    --condition_length 64 \
    --generation_length 256 \
    --save_images

# Inpainting（局部重新生成）
python inference.py \
    --checkpoint outputs/fractalgen/checkpoints/step_00100000.ckpt \
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

目前版本僅提供單一配置（約 30M 參數），對應 `768/384/192` 的層級嵌入維度與 `16/4/2/1` 的 Transformer block 數量，可在 8GB GPU 上以 `batch_size=8` 順利訓練。

## 🎛️ 重要參數

### 訓練參數

```python
--train_batch_size 8              # 訓練批次大小（單 GPU）
--val_batch_size 8                # 驗證批次大小
--max_steps 200000               # 總訓練步數
--val_check_interval_steps 2000  # 每隔多少步驗證一次
--checkpoint_every_n_steps 2000  # 每隔多少步儲存模型
--lr 1e-4                        # 學習率
--weight_decay 0.05              # 權重衰減
--warmup_steps 2000              # Warmup 步數
--accumulate_grad_batches 1      # 梯度累積步數
--grad_clip 3.0                  # 梯度裁剪
--devices 0,1                    # 使用的 GPU
--precision 32                   # 精度 (32/16/bf16)
--log_images_every_n_steps 5000  # 生成樣本頻率（0 關閉）
--cache_dir ./cache              # （選）piano roll 快取目錄
--no_cache_in_memory             # （選）停用記憶體快取
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

使用 `patch_size=4` 將 piano roll 劃分為 patches（pitch 維度固定為 128）:
- 128x512 → 4096 patches (32x32) [預設]
- 128x256 → 2048 patches (32x16)
- 128x128 → 1024 patches (32x8)

## 🎨 可視化

Piano roll 可視化遵循 Logic Pro 風格：
- **黑色**: 無音符 (velocity = 0)
- **綠色**: 中等力度 (velocity ≈ 64)
- **紅色**: 強力度 (velocity ≈ 127)

## 🔧 配置系統

### YAML 配置檔案

使用 YAML 檔案管理所有超參數，方便實驗管理：

```yaml
# config/train_default.yaml 範例

# 模型配置
model:
  generator_types: [mar, mar, mar, mar]
  scan_order: row_major
  mask_ratio_loc: 1.0
  mask_ratio_scale: 0.5
  grad_checkpointing: false

# 訓練配置
training:
  max_steps: 200000
  learning_rate: 1.0e-4
  weight_decay: 0.05
  warmup_steps: 2000
  grad_clip: 3.0
  accumulate_grad_batches: 1
  train_batch_size: 8
  val_batch_size: 8

# 數據配置
data:
  train_data: dataset/ariamidi/train.txt  # 使用 preprocess.py 生成的列表
  val_data: dataset/ariamidi/valid.txt
  crop_length: 512                        # 時間維度長度（128x512 piano roll）
  augment_factor: 1
  pitch_shift_min: -3
  pitch_shift_max: 3

# 硬體配置
hardware:
  devices: [0, 1]
  num_workers: 4
  precision: "32"

# 日誌配置
logging:
  output_dir: outputs/fractalgen
  val_check_interval_steps: 2000
  checkpoint_every_n_steps: 2000
  log_images_every_n_steps: 5000
```

### Dataclass 配置（內部）

程式碼內部使用 dataclass 管理配置：

```python
# trainer.py
@dataclass
class FractalTrainerConfig:
    max_steps: int = 200000
    grad_clip: float = 3.0
    accumulate_grad_batches: int = 1
    # ...

# dataset.py
@dataclass
class DataLoaderConfig:
    num_workers: int = 4
    pin_memory: bool = True
    # ...
```

YAML 配置會自動轉換為對應的 dataclass 實例。

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

### 推薦設置

```bash
# 使用配置檔案（推薦）
python main.py --config config/train_default.yaml

# 或使用命令行參數
python main.py \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --augment_factor 2 \
    --pitch_shift_min -3 \
    --pitch_shift_max 3 \
    --generator_types mar,mar,mar,mar \
    --max_steps 240000 \
    --val_check_interval_steps 2000 \
    --checkpoint_every_n_steps 2000 \
    --lr 1e-4 \
    --warmup_steps 4000 \
    --devices 0,1 \
    --precision 32 \
    --log_images_every_n_steps 0
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

- **限制**: 預設 `max_seq_len=2100`，可處理 128x256 左右的輸入；對於 128x512 需要更大的 `max_seq_len`
- **解決**: 如需更長序列（如 128x512），調整 `TrainerConfig.max_seq_len` 至少為 4096

### 3. 記憶體使用

- 約 8GB (batch_size=8)

## 📚 參考文件

### 文檔目錄

- **訓練指南**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- **模型結構**: [docs/MODEL_STRUCTURE.md](docs/MODEL_STRUCTURE.md)
- **GIF 生成**: [docs/GIF_GENERATION_GUIDE.md](docs/GIF_GENERATION_GUIDE.md)
- **Checkpoint 指南**: [docs/CHECKPOINT_MISMATCH_GUIDE.md](docs/CHECKPOINT_MISMATCH_GUIDE.md)

### 歷史文檔（存檔）

- [docs/archive/](docs/archive/)

### 原始參考

- **原論文**: https://arxiv.org/abs/2401.05036
- **原代碼**: https://github.com/Yikai-Liao/fractalgen

## 🐛 故障排除

### 訓練 Loss 不下降

```bash
# 檢查數據
python -c "
from dataset import create_dataloader, DataLoaderConfig
cfg = DataLoaderConfig.training_default('dataset/train.txt')
cfg.sampler.batch_size = 4
cfg.num_workers = 0
loader = create_dataloader(config=cfg)
batch = next(iter(loader))
print(f'Batch shape: {batch[0].shape}')
print(f'Value range: [{batch[0].min():.3f}, {batch[0].max():.3f}]')
if len(batch) > 2:
    print(f'Pitch shifts: {batch[2].tolist()}')
"

# 降低學習率
python main_fractalgen.py --lr 5e-5

# 增加 warmup
python main_fractalgen.py --warmup_steps 4000
```

### 記憶體不足

```bash
# 減小批次大小
python main_fractalgen.py --train_batch_size 2

# 啟用梯度檢查點（略降速度換取記憶體）
python main_fractalgen.py --grad_checkpoint

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

- 目前僅提供單一模型配置，建議先以短訓練驗證流程
- 監控 TensorBoard 的重建圖像質量與階層式指標

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

- 延長訓練: `--max_steps 400000`
- 數據增強（已內建隨機裁剪）
- 調整採樣參數

### 調試技巧

```bash
# 快速測試（1個 batch）
python main.py --config config/train_default.yaml --fast_dev_run

# 檢查模型
python -c "
from model import fractalmar_piano
model = fractalmar_piano()
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
"

# 檢查數據
python -c "
from dataset import create_dataloader, DataLoaderConfig
cfg = DataLoaderConfig.training_default('dataset/train.txt')
cfg.sampler.batch_size = 2
loader = create_dataloader(config=cfg)
for i, batch in enumerate(loader):
    print(f'Batch {i}: {batch[0].shape}')
    if i >= 2: break
"
```

## 🚀 下一步

### 短期（1-2週）

1. ✅ 完成模型訓練流程驗證
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
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - 訓練指南
- [docs/MODEL_STRUCTURE.md](docs/MODEL_STRUCTURE.md) - 模型結構
- [docs/](docs/) - 完整文檔目錄

