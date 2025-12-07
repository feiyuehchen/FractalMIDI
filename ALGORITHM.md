# 技術白皮書：Tempo-Aware Temporal Fractal Network (TFN) v2.0

> **摘要**：本文檔詳細介紹了 **FractalMIDI v2.0** 的架構、訓練動態和理論基礎。我們提出了一種分層生成模型，將音樂結構（速度、密度）與音樂內容（音符、力度）解耦。透過利用 **Adaptive Layer Normalization (AdaLN)** 進行條件控制、可配置的 **Harmonic Compression** 進行和聲傳遞、彈性的 **Flexible Bar Position Embedding** 強化節奏結構，以及改進的 **Deep Sampling** 優化生成細節，TFN 實現了 $O(\log L)$ 的推論複雜度，同時保持了標準自回歸 (AR) Transformer 通常難以實現的長期連貫性。

---

## 1. 問題陳述：為什麼「扁平」模型會失敗

標準音樂生成模型（例如 Music Transformer, MuseNet）將音樂視為扁平的 Token 序列 $x_1, x_2, ..., x_T$。這種方法有兩個致命缺陷：

1.  **缺乏層次規劃 (Lack of Hierarchical Planning)**：模型必須在每一步同時決定「下一個音符是什麼」和「全局速度是多少」。這導致了局部連貫但全局遊走（速度漂移、漫無目的的演奏）。
2.  **計算效率低 (Computational Inefficiency)**：生成長度為 $L$ 的歌曲需要 $L$ 個順序步驟（$O(L)$ 複雜度）。對於一首 3 分鐘的歌曲（$L \approx 4096$ tokens），這既慢又消耗記憶體（$O(L^2)$ attention）。

### 分形解決方案 (The Fractal Solution)
FractalMIDI 採用 **Level-of-Detail (LOD)** 方法，模仿人類作曲家的工作方式：
*   **草圖階段 (Sketch Phase)**：定義節奏、強度和結構。
*   **草稿階段 (Draft Phase)**：填充和弦和粗略旋律。
*   **潤飾階段 (Polish Phase)**：添加動態和裝飾音。

---

## 2. 資料處理與表示 (Data Processing & Representation)

資料處理流程由 `src/dataset/dataset.py` 與 `src/dataset/dataset_utils.py` 核心驅動，主要負責將原始 MIDI 文件轉換為模型可理解的 Tensor 格式。

### 2.1 鋼琴捲簾表示 (Piano Roll Representation)
模型不直接處理 MIDI 事件序列，而是處理多通道的鋼琴捲簾 (Piano Roll)。
*   **維度**: $(C, T, 128)$，其中 $C$ 為通道數，$T$ 為時間步，$128$ 為音高範圍。
*   **時間解析度**: 16 分音符 (1/16 note) 為一個時間步。透過 `symusic` 庫將 MIDI 以 `ticks_per_16th=120` 進行重採樣和量化。

### 2.2 通道定義 (Channels)
輸入資料包含三個主要通道：
1.  **Note On/Off (Channel 0)**: 二元網格 (Binary Grid)，1 表示有音符，0 表示休止。
2.  **Velocity (Channel 1)**: 連續值 $[0, 1]$，代表音符力度（響度）。
3.  **Tempo (Channel 2)**: 正規化曲線，代表 BPM 變化。
    *   計算方式：`Norm = (BPM - 40) / 160`，截斷於 $[0, 1]$。
4.  **Density (Derived)**: 每個時間步的音符密度（每秒音符數），歸一化至 $[0, 1]$，用於 Level 0 的結構引導。
5.  **Bar Position (New)**: 每個時間步在小節內的位置索引，用於水平結構化引導。在 v2.0 中，此嵌入不再限於固定的 4/4 拍，而是動態計算以適應不同的小節長度。

### 2.3 資料增強 (Data Augmentation)
為了提高模型的泛化能力，訓練過程中應用了以下增強技術：
*   **隨機裁切 (Random Crop)**: 將長序列隨機裁切為固定長度（預設 256 時間步）。
*   **移調 (Pitch Shift)**: 在 $[-3, +3]$ 半音範圍內隨機移動音高，讓模型學習相對音高關係而非絕對音高。

---

## 3. 模型架構與實作細節 (Model Architecture & Implementation Details)

FractalMIDI 採用遞歸的分形架構 (`src/models/temporal_fractal.py`)。本節將詳細說明其實作細節，包含張量維度與數據流。

### 3.1 分形層級定義 (Fractal Hierarchy)

模型將生成任務分解為 $N=3$ 個層級。假設我們生成一個 $128 \times 128$ 的片段（128 時間步，128 音高）：

| Level | 輸入解析度 | 下採樣因子 | 生成任務 | 核心組件 |
|:---|:---|:---|:---|:---|
| **Level 0** | Structure | 16 | 密度與速度 (Density & Tempo) | `TemporalGenerator` (Structure Mode) |
| **Level 1** | Content (Coarse) | 4 | 粗略音符與力度 | `TemporalGenerator` (Content Mode) |
| **Level 2** | Content (Fine) | 1 | 精細音符與力度 | `TemporalGenerator` (Content Mode) |

*   **遞歸邏輯**: Level $i$ 的生成結果會被上採樣並作為條件傳遞給 Level $i+1$。
*   **條件傳遞**: 引入了 **Harmonic Compression** 機制，將上一層的特徵壓縮後傳遞。

### 3.2 關鍵創新模組

#### A. 可配置的垂直壓縮 (Configurable Harmonic Compression)
為了提升跨層級的和聲連貫性並維持計算效率，我們將 Level $i$ 傳遞給 Level $i+1$ 的特徵進行了壓縮。
*   **實作**: 一個小型的 MLP Bottleneck，參數可在 `src/models/model_config.py` 中配置。
*   **公式**: $\mathcal{H}(\mathbf{F}_{L_i}) = \sigma(\mathbf{W}_h \cdot \mathbf{F}_{L_i})$，其中 $\sigma$ 可選 `relu`, `gelu` 或 `identity`。
*   **維度**: 從 `embed_dim` (例如 512) 壓縮到 `compressed_dim` (預設 32)。
*   **優勢**: 強迫模型學習最核心的和聲與結構特徵，去除噪聲，並減少下一層的參數負擔。這也為理論驗證提供了消融研究的空間。

#### B. 彈性水平結構化 (Flexible Bar Position Embedding)
將音樂的節奏結構（小節內的位置）作為一個強歸納偏置嵌入模型。
*   **輸入**: `bar_pos` 張量，形狀 $(B, T)$。
*   **實作**: 定義一個可學習的 Embedding 層 $\mathbf{E}_{\text{bar}}$，其大小由 `max_bar_len` 配置。
*   **整合**: $\mathbf{E}'_{\text{pos}} = \mathbf{E}_{\text{pos}} + \mathbf{E}_{\text{bar}}(\text{Bar Position})$
*   **優勢**: 使得模型在計算 Attention 時，天然地具備節奏結構的上下文，且不再受限於單一拍號，能處理更複雜的節奏結構。

#### C. 2D 音高卷積 (Pitch-Aware Convolution)
在 `FractalInputProj` 中，我們使用特殊的 2D 卷積核來處理輸入的鋼琴捲簾。
*   **核尺寸**: `(12, 1)`，即在音高軸上跨越 12 個半音（八度），在時間軸上為 1。
*   **目的**: 捕捉音程關係（如八度、五度），實現音高不變性 (Pitch Invariance)。

### 3.3 改進的 Deep Sampling (Refinement Loop)
在標準的生成過程之後，我們引入了一個迭代精煉步驟，以消除噪聲並增強局部一致性。
*   **流程**:
    1.  在完成主要的生成（Mask Ratio 1.0 $\to$ 0.0）後。
    2.  進入 Refinement Loop，迭代 $K$ 次（預設 3 次）。
    3.  每次迭代隨機遮罩 $p$ 的 Tokens (預設 30%，可配置)。
    4.  模型重新預測這些被遮罩的位置。
    5.  將預測結果更新回畫布：$\mathbf{x}_{k+1} = \mathbf{M} \odot \mathbf{\hat{x}}_k + (\mathbf{1} - \mathbf{M}) \odot \mathbf{x}_k$
*   **優勢**: 強制模型在微觀層次上自我修正，特別是在複雜的和聲轉換處。

---

## 4. 虛擬演算法與形狀變化詳解 (Algorithm Walkthrough & Shape Analysis)

本節詳細描述 Level 1 (Coarse Content) 的生成過程，並追蹤張量形狀的每一步變化。假設 Batch Size $B=4$，生成總長度 $L=256$，嵌入維度 $E=256$，音高數 $P=128$。

### Level 1: Coarse Content Generation

**1. 輸入準備 (Inputs)**
*   **Ground Truth**: 無（在 Inference 階段），初始化為全零或隨機噪聲。
*   **條件 (Condition)**: 來自 Level 0 (Structure) 的輸出，經過上採樣與壓縮。
*   **張量形狀**:
    *   `current_canvas`: $(B, 2, T_1, P) = (4, 2, 64, 128)$，其中 $T_1 = L / 4 = 64$。
    *   `prev_level_emb` (Level 0 Output): $(B, T_0, E_{L0}) = (4, 16, 512)$。

**2. 條件處理 (Condition Processing)**
*   **上採樣 (Upsample)**: 將 Level 0 的特徵在時間軸上插值到 Level 1 的解析度。
    *   $(B, 16, 512) \xrightarrow{\text{interpolate}} (B, 64, 512)$
*   **和聲壓縮 (Harmonic Compression)**: 通過 `cond_proj` 進行維度縮減。
    *   $(B, 64, 512) \xrightarrow{\text{Linear + Act}} (B, 64, 32)$
    *   結果 `level_cond`: $(B, 64, 32)$

**3. 輸入投影 (Input Projection)**
*   **2D 卷積**: 使用 Pitch-Aware Kernel (1, 12) 提取特徵。
    *   輸入 `current_canvas`: $(B, 2, 64, 128)$
    *   `FractalInputProj`: $(B, 2, 64, 128) \rightarrow (B, E, 64) = (4, 256, 64)$
*   **位置嵌入 (Positional Embeddings)**:
    *   時間位置: $+ \mathbf{E}_{\text{time}}(T)$
    *   小節位置 (Bar Pos): $+ \mathbf{E}_{\text{bar}}(\text{bar\_idx})$
    *   形狀維持: $(4, 256, 64)$

**4. Masked Auto-Regressive (MAR) Block**
*   **AdaLN Injection**: 將 `level_cond` 注入到每個 Transformer Block。
    *   `x`: $(B, 64, 256)$, `cond`: $(B, 64, 32)$
    *   $\text{AdaLN}(x, c) = \gamma(c) \cdot \frac{x - \mu}{\sigma} + \beta(c)$
*   **Self-Attention (FlashAttention)**:
    *   計算 $Q, K, V$: $(B, 64, 256)$
    *   Attention Map (Implicit): $(B, 64, 64)$
    *   輸出: $(B, 64, 256)$
*   **MLP**: Feed-forward network。
    *   輸出: $(B, 64, 256)$

**5. 輸出投影 (Output Projection)**
*   **反卷積/線性投影**: 將特徵映射回音高空間。
    *   `FractalOutputProj`: $(B, 256, 64) \rightarrow (B, 2 \times 128, 64)$
    *   Reshape: $\rightarrow (B, 2, 64, 128)$
*   **Logits**:
    *   Channel 0 (Note On/Off): Sigmoid $\rightarrow [0, 1]$
    *   Channel 1 (Velocity): ReLU/Clamp $\rightarrow [0, 1]$

**6. 迭代更新 (Iterative Update)**
*   對於 MAR 採樣，我們進行 $K$ 次迭代。
*   每次迭代根據 Mask Ratio 重新採樣部分位置，並將預測結果填回 `current_canvas`。
*   最終輸出 `final_output`: $(B, 2, 64, 128)$

---

## 5. 訓練細節 (Training Details)

### 5.1 訓練流程 (Training Process)
模型的 `forward` 函數是遞歸定義的，訓練時一次性計算所有層級的 Loss：
1.  **下採樣 GT**: 將原始 MIDI (`notes`, `tempo`, `density`) 下採樣至各層級解析度。
2.  **隨機遮罩**: 對每一層隨機採樣 `mask_ratio` (0.5 ~ 1.0)，生成二元 Mask。
3.  **層級前向**:
    *   Level 0: 輸入 masked GT，預測密度與速度。
    *   Level 1+: 輸入 masked GT 和上一層的 GT Embedding (Teacher Forcing)，預測音符。
4.  **損失聚合**: 加總所有層級的 Loss。

### 5.2 損失函數 (Loss Functions)
*   **Structure Level (Level 0)**:
    *   Density: MSE Loss
    *   Tempo: MSE Loss
*   **Content Level (Level 1+)**:
    *   Note: Binary Cross Entropy with Logits Loss (加權 10.0)
    *   Velocity: MSE Loss (僅計算有音符的位置，加權 1.0)

### 5.3 訓練動態與超參數 (Hyperparameters)
採用 **Two-Stage Warmup** 策略以穩定混合訓練：

1.  **Phase 1 (0 - 5k steps): "Learn to Speak"**
    *   `Tempo/Structure Weight` = **0.0**
    *   目標：強迫梯度專注於音符位置的準確性，避免模型只學會輸出平均速度而忽略音符。
2.  **Phase 2 (5k+ steps): "Learn to Flow"**
    *   `Tempo/Structure Weight` = **1.0**
    *   目標：在音符生成穩定的基礎上，引入結構一致性。

**關鍵超參數**:
*   **Optimizer**: AdamW
    *   `lr`: 1e-4
    *   `betas`: (0.9, 0.95)
    *   `weight_decay`: 0.01
*   **Batch Size**: 16 (per GPU)
*   **Sequence Length**: 256 (random crop)
*   **Precision**: BF16 (Mixed Precision)

---

## 6. 推論與評估 (Inference & Evaluation)

### 6.1 取樣策略 (Sampling Strategy)
*   **Masked Auto-Regressive (MAR) Sampling**:
    *   迭代式生成：從隨機噪聲開始，預測一部分，重新 Mask，再預測。
*   **Deep Sampling**:
    *   在初步生成後進行多次 Refinement，提升品質。
*   **Inpainting**:
    *   支援鎖定特定區域（通過 Mask），只重生成選定範圍，同時保持與周圍環境（結構和內容）的連貫性。

### 6.2 評估方法 (Evaluation)
v2.0 引入了客觀的定量評估指標 (`src/inference/metrics.py`)：
*   **Pitch Entropy (音高熵)**: 測量音高分佈的複雜度。
*   **Scale Consistency (調性一致性)**: 計算音符符合主要調性的比例。
*   **Groove Consistency (律動一致性)**: 評估節奏模式在小節間的穩定性。

此外仍保留多模態的定性分析：
*   **視覺化 (Visual)**: 生成 GIF 動畫展示分形生成過程，以及靜態的鋼琴捲簾熱圖 (PNG)。
*   **聽覺化 (Audio)**: 將生成的 Tensor 轉換回 MIDI 文件進行試聽。
*   **條件影響視覺化**: 顯示模型在生成過程中關注的條件區域。

---

## 7. 使用者介面與系統集成 (UI & System Integration)

系統提供了一個完整的 Web 介面，位於 `src/web/` 目錄下。

### 7.1 技術堆疊
*   **Backend**: Python **FastAPI** (`src/web/backend/app.py`)。
*   **Frontend**: HTML/JS/CSS (`src/web/frontend/static/`)，無複雜框架，輕量級實現。
*   **通信**: 
    *   REST API 用於模型管理、範例加載。
    *   **WebSocket** (`/ws/generate`) 用於實時 Fractal Streaming，將中間生成步驟流式傳輸到前端。

### 7.2 功能特性
*   **分形流式傳輸 (Fractal Streaming)**: 用戶可以看到音樂從模糊的熱圖（Level 0）逐漸變清晰（Level 1+）的過程。
*   **TouchDesigner Bridge**: 提供專用的 WebSocket 端點 (`/ws/touchdesigner`)，將生成的音符實時發送給 TouchDesigner 進行視覺藝術創作。
*   **條件可視化**: 前端能夠顯示當前生成受哪些條件影響。

---

## 8. 複雜度分析 (Complexity Analysis)

| 特性 | 標準 AR Transformer | FractalMIDI v2 |
|:---|:---|:---|
| **時間複雜度** | $O(L)$ (Sequential) | $O(\log L)$ (Parallel Hierarchical) |
| **空間複雜度** | $O(L^2)$ (Standard Attn) | $O(L)$ (FlashAttention) |
| **歸納偏置** | 弱 (從頭學習) | 強 (音高感知 + 分層結構 + 節奏感知) |
| **速度感知** | 隱式 (難以控制) | 顯式 (通過 AdaLN + Bar Embedding) |

FractalMIDI v2 證明了通過引入強大的音樂歸納偏置（層級性、音高不變性、節奏結構），我們可以比暴力破解的大型語言模型獲得更好的性能和可控性。
