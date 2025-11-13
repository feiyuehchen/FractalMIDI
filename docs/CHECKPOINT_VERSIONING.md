# Checkpoint 版本管理

## 概述

從現在開始，訓練的 checkpoints 和 logs 都使用相同的版本號進行管理，每次訓練會自動創建新版本。

## 目錄結構

```
outputs/
└── fractalgen/              # 實驗基礎目錄
    ├── checkpoints/         # Checkpoint 版本管理
    │   ├── version_0/
    │   │   ├── config.yaml                      # 訓練配置
    │   │   ├── step_00010000-val_loss_0.1234.ckpt
    │   │   ├── step_00020000-val_loss_0.1123.ckpt
    │   │   └── ...
    │   ├── version_1/
    │   │   ├── config.yaml
    │   │   └── ...
    │   └── version_2/
    │       └── ...
    └── logs/                # TensorBoard 日誌
        ├── version_0/
        ├── version_1/
        └── version_2/
```

## 特性

### 1. 自動版本控制

每次運行訓練時，系統會：
- 自動檢測現有的版本號
- 創建新的 `version_N` 目錄
- Checkpoints 和 logs 使用**相同的版本號**

### 2. 配置文件保存

每個 checkpoint 版本目錄都包含 `config.yaml`：

**如果使用配置文件訓練：**
```bash
python main.py --config config/train_default.yaml
```
→ 原始配置文件會被複製到 `checkpoints/version_N/config.yaml`

**如果使用命令行參數訓練：**
```bash
python main.py --train_batch_size 8 --max_steps 100000
```
→ 系統會自動生成 `config.yaml` 包含所有使用的參數

### 3. 版本號同步

版本號在 checkpoints 和 logs 之間同步：
- `checkpoints/version_0/` ↔ `logs/version_0/`
- `checkpoints/version_1/` ↔ `logs/version_1/`

這樣可以輕鬆找到對應的訓練記錄！

## 使用範例

### 開始新的訓練

```bash
# 使用配置文件
python main.py --config config/train_small.yaml
```

輸出：
```
✓ Experiment version: 0
✓ Checkpoint dir: outputs/fractalgen_small/checkpoints/version_0
✓ Saved config from: config/train_small.yaml
✓ Config saved to: outputs/fractalgen_small/checkpoints/version_0/config.yaml
```

### 查看訓練配置

```bash
# 查看某個版本的配置
cat outputs/fractalgen/checkpoints/version_0/config.yaml

# 使用相同配置重新訓練
python main.py --config outputs/fractalgen/checkpoints/version_0/config.yaml
```

### 找到對應的 TensorBoard 日誌

```bash
# Checkpoint 在 version_2
ls outputs/fractalgen/checkpoints/version_2/

# 對應的 logs 也在 version_2
tensorboard --logdir outputs/fractalgen/logs/version_2
```

## 推理時使用

使用帶版本的 checkpoint：

```bash
# 使用特定版本的 checkpoint
python inference.py \
    --checkpoint outputs/fractalgen/checkpoints/version_0/step_00050000-val_loss_0.1234.ckpt \
    --mode unconditional
```

可以查看對應的配置：
```bash
cat outputs/fractalgen/checkpoints/version_0/config.yaml
```

## 優點

1. **可追溯性** - 每個 checkpoint 都有完整的訓練配置記錄
2. **可重現性** - 使用保存的 config 可以精確重現訓練
3. **組織性** - 版本號使多次實驗易於管理
4. **同步性** - Checkpoints 和 logs 版本對應，不會混淆

## 與舊系統的區別

| 特性 | 舊系統 | 新系統 |
|------|--------|--------|
| Checkpoint 目錄 | `checkpoints/` (平級) | `checkpoints/version_N/` |
| 配置保存 | ❌ 無 | ✅ 自動保存 config.yaml |
| 版本管理 | ❌ 只有 logs 分版本 | ✅ Checkpoints + logs 都分版本 |
| 版本對應 | ❌ 需手動對應 | ✅ 自動同步版本號 |

## 注意事項

- 版本號由 TensorBoardLogger 自動管理
- 刪除 `logs/version_N` 會影響版本號計算
- 建議保持 checkpoints 和 logs 目錄結構同步
- 配置文件使用 YAML 格式，易於閱讀和編輯

