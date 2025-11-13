# 配置系統重構指南

## 概述

將所有硬編碼的參數移至統一的配置系統，提高代碼可維護性和靈活性。

## ✅ 已完成

### 1. 創建配置數據類 (`models/model_config.py`)

新建了完整的配置結構：

```python
@dataclass
class PianoRollConfig:
    height: int = 128                    # MIDI pitch range  
    max_width: int = 512                 # Maximum time steps
    patch_size: int = 4                  # Patch size
    velocity_vocab_size: int = 256       # MIDI velocity [0-255]

@dataclass  
class ArchitectureConfig:
    img_size_list: Tuple[int, ...] = (128, 16, 4, 1)
    embed_dim_list: Tuple[int, ...] = (512, 256, 128, 64)
    num_blocks_list: Tuple[int, ...] = (12, 3, 2, 1)
    num_heads_list: Tuple[int, ...] = (8, 4, 2, 2)
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    init_std: float = 0.02
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6

@dataclass
class GeneratorConfig:
    generator_type_list: Tuple[str, ...] = ("mar", "mar", "mar", "mar")
    scan_order: str = "row_major"
    mask_ratio_loc: float = 1.0
    mask_ratio_scale: float = 0.5
    num_conds: int = 5

@dataclass
class TrainingConfig:
    grad_checkpointing: bool = False
    v_weight: float = 1.0

@dataclass
class FractalModelConfig:
    piano_roll: PianoRollConfig
    architecture: ArchitectureConfig
    generator: GeneratorConfig
    training: TrainingConfig
```

###  2. 部分更新 `models/fractal_gen.py`

- ✅ 添加 `model_config` 參數支持
- ✅ 保持向後兼容性（可使用舊的單獨參數）
- ✅ 更新初始化參數使用 `config.architecture.init_std`
- ✅ 更新序列長度計算使用 `config.piano_roll.height` 和 `config.piano_roll.max_width`
- ✅ 更新 generator_kwargs 使用配置值
- ✅ 更新遞歸調用傳遞配置
- ✅ 更新 sample() 方法使用配置

## ⚠️ 待完成

### 3. 更新 `models/velocity_loss.py`

**當前硬編碼**：
```python
self.v_codebook = nn.Embedding(256, width)  # Line 38
self.v_mlm = MlmLayer(256)  # Line 50
```

**需要修改**：
```python
def __init__(self, ..., velocity_vocab_size=256):
    self.v_codebook = nn.Embedding(velocity_vocab_size, width)
    self.v_mlm = MlmLayer(velocity_vocab_size)
```

### 4. 更新 `models/mar_generator.py`

**當前硬編碼**：
```python
# Line 23
def __init__(self, ..., img_size=128, ...):

# Line 185
h = 128 // self.patch_size  # Known height
```

**需要修改**：
```python
def __init__(self, ..., img_size=128, piano_roll_height=128, ...):
    self.piano_roll_height = piano_roll_height
    
# Later:
h = self.piano_roll_height // self.patch_size
```

### 5. 更新 `models/ar_generator.py`

**當前硬編碼**：
```python
# Line 19
def __init__(self, ..., img_size=128, ...):

# Line 24
if self.img_size >= 128:
```

**需要修改**：
```python
def __init__(self, ..., img_size=128, piano_roll_height=128, ...):
    self.piano_roll_height = piano_roll_height
    
# Later: 
if self.img_size >= self.piano_roll_height:
```

### 6. 更新 `models/generation.py`

**當前硬編碼**：
```python
# Lines 15, 23, 34, 59, 70, 79, 89, 117
# 所有地方都硬編碼了 128 (piano roll height)
full_roll = torch.zeros(1, 1, 128, padded_length, device=device)
```

**需要修改**：
```python
def conditional_generation(model, condition_roll, generation_length, piano_roll_height=None, ...):
    if piano_roll_height is None:
        piano_roll_height = model.config.piano_roll.height
    full_roll = torch.zeros(1, 1, piano_roll_height, padded_length, device=device)
```

### 7. 更新 `trainer.py` 的 ModelConfig

需要添加新的配置字段：

```python
@dataclass
class ModelConfig:
    # Piano roll settings
    piano_roll_height: int = 128
    max_crop_length: int = 512
    patch_size: int = 4
    velocity_vocab_size: int = 256
    
    # Architecture
    img_size_list: Tuple[int, int, int, int] = (128, 16, 4, 1)
    embed_dim_list: Tuple[int, int, int, int] = (512, 256, 128, 64)
    num_blocks_list: Tuple[int, int, int, int] = (12, 3, 2, 1)
    num_heads_list: Tuple[int, int, int, int] = (8, 4, 2, 2)
    
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    init_std: float = 0.02
    mlp_ratio: float = 4.0
    
    # Generator
    generator_type_list: Tuple[str, str, str, str] = ('mar', 'mar', 'mar', 'mar')
    scan_order: str = 'row_major'
    mask_ratio_loc: float = 1.0
    mask_ratio_scale: float = 0.5
    num_conds: int = 5
    
    # Training
    grad_checkpointing: bool = False
    v_weight: float = 1.0
    
    def to_fractal_config(self):
        """Convert to FractalModelConfig"""
        from models.model_config import (
            FractalModelConfig, PianoRollConfig, 
            ArchitectureConfig, GeneratorConfig, TrainingConfig
        )
        return FractalModelConfig(
            piano_roll=PianoRollConfig(
                height=self.piano_roll_height,
                max_width=self.max_crop_length,
                patch_size=self.patch_size,
                velocity_vocab_size=self.velocity_vocab_size,
            ),
            architecture=ArchitectureConfig(
                img_size_list=self.img_size_list,
                embed_dim_list=self.embed_dim_list,
                num_blocks_list=self.num_blocks_list,
                num_heads_list=self.num_heads_list,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
                init_std=self.init_std,
                mlp_ratio=self.mlp_ratio,
            ),
            generator=GeneratorConfig(
                generator_type_list=self.generator_type_list,
                scan_order=self.scan_order,
                mask_ratio_loc=self.mask_ratio_loc,
                mask_ratio_scale=self.mask_ratio_scale,
                num_conds=self.num_conds,
            ),
            training=TrainingConfig(
                grad_checkpointing=self.grad_checkpointing,
                v_weight=self.v_weight,
            )
        )
```

### 8. 更新 YAML 配置文件

在現有的 YAML 配置中添加新字段（可選，已有字段會使用默認值）：

```yaml
model:
  # Piano roll settings (NEW)
  piano_roll_height: 128
  patch_size: 4
  velocity_vocab_size: 256
  
  # Architecture (EXISTING + NEW)
  img_size_list: [128, 16, 4, 1]
  embed_dim_list: [512, 256, 128, 64]
  num_blocks_list: [12, 3, 2, 1]
  num_heads_list: [8, 4, 2, 2]
  attn_dropout: 0.1                    # NEW
  proj_dropout: 0.1                    # NEW
  init_std: 0.02                       # NEW
  mlp_ratio: 4.0                       # NEW
  
  # Generator (EXISTING + NEW)
  generator_types: [mar, mar, mar, mar]
  scan_order: row_major
  mask_ratio_loc: 1.0
  mask_ratio_scale: 0.5
  num_conds: 5                         # NEW
  
  # Training (EXISTING + NEW)
  grad_checkpointing: true
  v_weight: 1.0                        # NEW
```

## 📝 遷移步驟

### 短期（保持向後兼容）

1. ✅ 創建 `models/model_config.py`
2. ✅ 更新 `models/fractal_gen.py` 支持雙模式（config 對象或單獨參數）
3. ⚠️ 更新 `models/velocity_loss.py` 添加參數
4. ⚠️ 更新 `models/mar_generator.py` 添加參數
5. ⚠️ 更新 `models/ar_generator.py` 添加參數
6. ⚠️ 更新 `models/generation.py` 添加參數
7. ⚠️ 更新 `trainer.py` 的 ModelConfig
8. ⚠️ 測試所有功能正常

### 長期（完全遷移）

1. 全面使用 `FractalModelConfig` 對象
2. 移除對單獨參數的支持
3. 簡化代碼

## 🎯 優點

### 1. **消除魔術數字**
- 所有參數都有明確的名稱和文檔
- 易於理解和修改

### 2. **集中管理**
- 所有配置在一個地方定義
- 避免不一致性

### 3. **類型安全**
- 使用 dataclass 提供類型檢查
- `__post_init__` 驗證參數有效性

### 4. **靈活性**
- 易於創建不同的配置預設
- 支持從 YAML 加載/保存

### 5. **向後兼容**
- 現有代碼仍可工作
- 漸進式遷移

## 🔍 示例用法

### 使用新配置系統

```python
from models.model_config import FractalModelConfig, PianoRollConfig

# 使用默認配置
config = FractalModelConfig()

# 自定義配置
config = FractalModelConfig(
    piano_roll=PianoRollConfig(
        height=128,
        max_width=256,
    ),
)

# 創建模型
model = PianoRollFractalGen(model_config=config)
```

### 向後兼容方式（舊代碼）

```python
# 仍然可以使用舊的方式
model = PianoRollFractalGen(
    img_size_list=(128, 16, 4, 1),
    embed_dim_list=(512, 256, 128, 64),
    # ... 其他參數
)
```

## 📊 硬編碼清單

### 已消除
- ✅ `piano_roll_height = 128` in fractal_gen.py
- ✅ `max_crop_length = 512` in fractal_gen.py  
- ✅ `init_std = 0.02` in fractal_gen.py
- ✅ `attn_dropout = 0.1` in fractal_gen.py
- ✅ `proj_dropout = 0.1` in fractal_gen.py
- ✅ `mask_ratio_loc = 1.0` in fractal_gen.py
- ✅ `mask_ratio_scale = 0.5` in fractal_gen.py
- ✅ `num_conds = 5` in fractal_gen.py
- ✅ `target_width = 256` default in sample()

### 待消除
- ⚠️ `velocity_vocab_size = 256` in velocity_loss.py
- ⚠️ `piano_roll_height = 128` in mar_generator.py (line 185)
- ⚠️ `piano_roll_height = 128` in ar_generator.py  
- ⚠️ `piano_roll_height = 128` in generation.py (多處)
- ⚠️ Various dropout/mlp_ratio in blocks.py, attention.py

## 🚀 下一步

1. 完成待辦事項中的文件更新
2. 更新所有 YAML 配置文件添加新字段
3. 運行測試確保一切正常
4. 更新文檔反映新的配置系統
5. 考慮移除舊的單獨參數方式（向後不兼容的變更）

