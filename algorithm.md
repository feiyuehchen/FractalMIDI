# FractalMIDI 演算法技術文檔

## 1. 總覽
FractalMIDI 是一個階層式的符號音樂（Symbolic Music, Piano Roll）生成模型，旨在生成連貫、結構嚴謹且富含表現力的音樂。它透過多個時間解析度（層級）進行操作，首先生成音樂的整體結構，然後遞迴地細化每個部分的細節。

## 2. 核心架構：時間碎形網路 (Temporal Fractal Network, TFN)

本模型建立在多層級的 **時間生成器 (Temporal Generators)** 之上，每個生成器負責特定的時間解析度。

-   **層級 (Levels)**：
    -   **Level 0 (結構層)**：生成全域屬性，如 **音符密度 (Note Density)** 和 **速度 (Tempo)** 曲線。
    -   **Level 1+ (內容層)**：生成實際的 **鋼琴捲簾 (Piano Roll)**（包含音符開關與力度），並以前一層的輸出作為條件。

### 2.1. 混合生成策略 (Hybrid Generation Strategy)
本模型採用混合生成策略，結合了雙向上下文 (Bidirectional Context) 與自回歸 (Autoregressive) 的優勢：

-   **時間軸 (Time-Axis) - MAR**：採用 **遮罩自回歸 (Masked Auto-Regression, MAR)** 邏輯。模型根據雙向上下文（過去和未來）來預測被遮罩的區域。這確保了全域結構的連貫性（例如樂句的起承轉合）。
-   **音高軸 (Pitch-Axis) - AR**：在每一個時間步 (Time Step) 內，音高的生成採用 **自回歸 (Autoregressive, AR)** 解碼。這確保了和聲的一致性（生成合理的和弦與音程關係）。

### 2.2. 編碼器：ConvNeXt V2 + Attention
輸入投影層 (`FractalInputProj`) 已全面升級，以從鋼琴捲簾中提取更強健的特徵：

1.  **Stem 層**：使用 `Conv2d` 降低音高維度 (128 -> 32) 但保留完整的時間解析度 ($T$)。
2.  **骨幹網路 (Backbone)**：使用兩個 **ConvNeXt V2 Block**（包含深度卷積 Depthwise Conv、LayerNorm、GELU、GRN），處理 (Pitch, Time) 特徵圖以捕捉局部的音樂紋理。
3.  **投影 (Projection)**：將特徵展平並投影到嵌入維度 ($E$)。
4.  **位置編碼 (Positional Embedding)**：加入 **多尺度時間位置編碼 (Multi-Scale Time PE)** 與小節編碼 (Bar PE)。
5.  **全域上下文**：使用 **Attention Block** (GPT-OSS 風格，含 RMSNorm 與 SwiGLU) 來整合全域的時間資訊。

### 2.3. 解碼器：Pitch Transformer + RoPE
輸出投影層 (`PitchTransformerOutputProj`) 是一個專門用於音高生成的 Transformer Decoder：

-   **自回歸生成**：針對每個時間步，依序生成 128 個音高 (MIDI Pitch 0-127)。
-   **旋轉位置編碼 (RoPE)**：應用於音高序列，顯式地建模相對音程關係（例如：移調不變性、和弦結構）。
-   **教師強制 (Teacher Forcing)**：在訓練期間，接收 Ground Truth 目標以進行並行化訓練。

### 2.4. 位置編碼 (Positional Embeddings)
-   **多尺度絕對時間編碼 (Multi-Scale Absolute Time PE)**：透過加總四個尺度的嵌入向量來捕捉階層化的時間結構：
    -   小節索引 (Bar Index, $t \pmod{16}$)
    -   四分音符索引 (Quarter Note Index, $t \pmod{4}$)
    -   8 小節樂句索引 (8-Bar Phrase Index, $t \pmod{128}$)
    -   全域絕對索引 (Global Absolute Index, $t$)
-   **旋轉位置編碼 (RoPE)**：用於 Pitch Decoder 以捕捉相對音高距離。

## 3. 訓練流程

### 3.1. 和聲條件 (Harmonic Conditioning - Chroma)
-   **特徵**：從資料集中提取 12 維的 **Chroma (Pitch Class Profile)** 向量，代表隨時間變化的和聲內容。
-   **條件化**：此 Chroma 向量作為全域條件 (`global_cond`) 傳遞給模型的所有層級，引導生成過程遵循特定的和聲進行。
-   **無分類器引導 (CFG)**：在訓練期間應用 Classifier-Free Guidance（以 10% 機率將條件設為零），以便在推論時進行無條件或引導式採樣。

### 3.2. 損失函數 (Loss Function)
-   **結構層 (Structure Level)**：使用 MSE Loss 計算密度與速度曲線的誤差。
-   **內容層 (Content Level)**：
    -   **音符開關 (Note On/Off)**：使用二元交叉熵 (BCE) Loss。
    -   **力度 (Velocity)**：使用 MSE Loss（僅計算有音符的位置）。

### 3.3. 採樣 (Sampling)
-   **溫度控制 (Temperature Control)**：使用溫度參數來調節 Logits 的分佈。
    -   **低溫**：生成更確定、和諧且重複性高的模式。
    -   **高溫**：生成更多樣化且具表現力的模式。
    -   **伯努利採樣 (Bernoulli Sampling)**：最終音符的選擇基於機率進行伯努利採樣。

## 4. 資料流程
-   **資料集**：Pop909 資料集（預處理為鋼琴捲簾格式）。
-   **資料增強 (Augmentation)**：
    -   **隨機裁切 (Random Crop)**：隨機裁切序列至固定長度（如 512）。
    -   **移調 (Pitch Shift)**：隨機對音樂及對應的 Chroma 特徵進行移調。
-   **批次處理**：使用桶裝採樣 (Bucket Sampling) 將長度相近的序列分組（若未啟用裁切），以提高效率。

## 5. 近期改進摘要
1.  **Encoder 升級**：將簡單的 CNN 替換為 **ConvNeXt V2 + GPT-OSS Attention**，提升特徵提取能力。
2.  **Decoder 升級**：將 LSTM/Linear 解碼器替換為 **Pitch Transformer + RoPE**，增強和聲建模。
3.  **混合 MAR/AR**：在 MAR 模式中啟用 Decoder，確保和弦的一致性。
4.  **多尺度 Time PE**：加入顯式的小節/拍子/樂句結構建模。
5.  **視覺化修復**：修正驗證集的視覺化邏輯，正確進行上採樣並使用折線圖顯示結構層。
6.  **訓練穩定性**：修正 AR 訓練中的 Teacher Forcing 邏輯並微調採樣參數。
