# 待解決問題與未來展望 (Issues & Future Work)

## 1. 推論速度 (Inference Speed)
-   **問題**：目前的 Decoder 需要優化以提升推論效率。
-   **現狀**：尚未實作 KV Cache，且架構需要明確定義以支持高效生成。
-   **改進方向**：
    -   **架構調整**：不要使用 LSTM，改用 2 層、1024 維度 (dim)、單一 Head (head1) 的 Transformer Block。
    -   **機制實作**：參考 `fractalgen/models/ar.py` 中的 `KVCache` 實作方式，將其移植到 `PitchTransformerOutputProj`，以加速自回歸生成。

## 2. 訓練穩定性與平衡 (Training Stability)
-   **問題**：模型同時包含 MAR (Level 0, 1) 和 AR (Level 2, 3) 模組，多任務 Loss 權重需要精細調整。
-   **改進方向**：
    -   **相關論文**：
        -   *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall et al., 2018)*
        -   **更新 (2024)**：可參考 *GradNorm* 或 *Dynamic Weight Averaging (DWA)* 等更先進的平衡策略，解決不同任務梯度量級差異過大的問題。
    -   **實作策略**：引入基於不確定性 (Uncertainty) 的動態權重調整 (Dynamic Loss Weighting)。
    -   **演算法步驟**：
        1.  **定義可學習參數**：為每個任務 $i$ (Level 0~3) 定義一個可學習的參數 $\sigma_i$ (通常實作為 $s_i := \log \sigma_i^2$ 以保證數值穩定)。
        2.  **修正 Loss 函數**：將總 Loss 定義為 $L_{total} = \sum_i (\frac{1}{2\sigma_i^2} L_i + \log \sigma_i)$。
        3.  **聯合優化**：在訓練過程中，Optimizer 同時更新模型權重與任務權重 $\sigma_i$。
        4.  **機制原理解析**：當某個任務 $L_i$ 較高時（不確定性高），模型會傾向增加 $\sigma_i$ 來降低該項的權重，但 $\log \sigma_i$ 正則項會懲罰過大的 $\sigma_i$，迫使模型在兩者間找到平衡點，而非單純忽略困難任務。
    -   **潛在風險**：
        -   **權重坍塌 (Weight Collapse)**：在極端情況下，模型可能會無止境地增加某個困難任務的 $\sigma_i$ 來「作弊」降低 Loss，導致該任務完全被忽略。需設置 $\sigma_i$ 的上下限或正則項。
        -   **優化不穩定**：$\sigma$ 也是可學習參數，若學習率設定不當，可能導致 Loss 震盪劇烈，難以收斂。

## 3. 數據集規模與模型容量 (Data & Capacity)
-   **問題**：目前主要依賴 Pop909 資料集。
-   **改進方向**：
    -   **優先目標**：雖然可以加入 Maestro 等更多資料，但目前首要目標是先在現有數據上觀察到**過擬合 (Overfitting)** 的效果，以驗證模型容量 (Capacity) 足夠捕捉複雜特徵。
    -   **後續步驟**：確認模型能力後，再整合更多 MIDI 資料集並增強資料增強策略。

## 4. 表現力與力度控制 (Expressiveness)
-   **問題**：目前的力度 (Velocity) 生成使用 MSE Loss，導致動態變化平淡。
-   **改進方向**：
    -   **相關論文**：
        -   *Performance RNN (Oore et al., 2018)* 與 *Music Transformer (Huang et al., 2018)* (離散化)
        -   **更新 (2023-2024)**：可參考 *Museformer* 或擴散模型 (Diffusion Models) 如 *SongBloom* (2025)，它們在細微特徵的生成上表現更佳，能產生更具人性化的力度變化。
    -   **決策流程**：先研讀上述論文關於 32 或 64 個等級分類預測的效果，再決定具體的改動方式。
    -   **演算法步驟**：
        1.  **量化 (Quantization)**：將 0-127 的連續力度值映射到 $N$ 個離散區間（Bin）。例如 $N=32$，則 Bin 0 代表力度 0-3，Bin 1 代表 4-7，依此類推。
        2.  **模型輸出調整**：將力度預測層的輸出從單一數值改為 $N$ 維 logits (Softmax)。
        3.  **Loss 轉換**：使用 Cross Entropy Loss 取代 MSE Loss。
        4.  **取樣策略**：在推論時，依據預測的機率分佈進行採樣 (Sampling)，而非僅取最大值或平均值。這允許模型生成偶爾出現的「重音」或「鬼音 (Ghost Notes)」，而非總是趨向平均力度，從而增加演奏的動態感與人性化。
    -   **潛在風險**：
        -   **量化誤差**：離散化必定會丟失精度，可能導致漸強/漸弱 (Crescendo/Diminuendo) 的平滑度下降，聽感上出現階梯狀突變。

## 5. 長期結構依賴 (Long-term Structure)
-   **問題**：長期結構依賴需要更大的時間步 (Time Step) 和上下文視窗，目前的 Transformer Block 在長序列下效率不足。
-   **改進方向**：
    -   **策略**：**不增加新的層級 (Level -1)**，而是直接升級現有 **Level 0~3** 的 `TemporalGenerator` 骨幹網路。
    -   **機制選擇**：優先採用 **Music Transformer** (Relative Attention) 及其變體（如 **Museformer**）。(註：Mamba/SSM 雖然高效，但考慮到訓練穩定性與實作複雜度，暫不採用)。
    -   **實作位置**：在 `src/models/temporal_fractal.py` 的 `FractalBlock` 中進行替換。
    -   **更新 (2022-2024)**：除了標準的 Relative Attention，可參考 **Museformer (2022)** 提出的 *Fine- and Coarse-Grained Attention*，它結合了對近期小節的精細 Attention 與對遠期小節的摘要 Attention，能將序列長度擴展 3 倍以上且計算更高效。
    -   **演算法步驟 A: Music Transformer (Relative Global Attention)**：
        1.  **相對位置嵌入**：引入相對位置編碼 $E^r$，計算 Attention 分數時包含內容-位置交互項 $S_{rel} = Q E^r^T$。
        2.  **Skewing 機制 (記憶體優化)**：
            -   計算 $QE^r^T$ 得到形狀為 $(L, L)$ 的矩陣。
            -   在矩陣左側填充一列 $0$，形狀變為 $(L, L+1)$。
            -   將矩陣 Reshape 為 $(L+1, L)$。
            -   切片取前 $L$ 列，並移除超出下三角的部分（Masking）。
            -   **目的**：此操作將相對距離 $i-j$ 索引對齊到絕對位置 $(i, j)$，避免了存儲 $O(L^2 \times D)$ 的中間張量，使長序列訓練成為可能。
    -   **潛在風險**：
        -   **記憶體與速度權衡**：Relative Attention 雖然優化了記憶體，但計算量仍是 $O(L^2)$，在極長序列下仍有瓶頸（可考慮配合 Museformer 的稀疏機制緩解）。

## 6. 評估指標 (Evaluation)
-   **問題**：缺乏客觀的音樂品質指標。
-   **改進方向**：
    -   **新增指標**：參考 *The Jazz Transformer (Wu & Yang, 2020)* 及 *MusDr* 專案的指標。
    -   **演算法步驟 (MusDr Metrics)**：
        1.  **Pitch-Class Histogram Entropy (H)**：
            -   將樂曲切分為短片段（如 1 或 4 小節）。
            -   計算每段的音高類別 (Pitch Class, 0-11) 直方圖並歸一化為機率分佈。
            -   計算該分佈的熵 (Entropy)。**高熵**代表音高使用混亂（隨機），**低熵**代表調性明確。
        2.  **Grooving Pattern Similarity (GS)**：
            -   提取每小節的節奏模式（Grooving Pattern），通常為二值化的 Onset 向量。
            -   計算所有小節對 $(i, j)$ 之間的相似度（如 Cosine Similarity）。
            -   取平均值。**高 GS** 代表整首曲子的節奏風格一致且穩定。
        3.  **Chord Progression Irregularity (CPI)**：
            -   提取曲子的和弦進行序列。
            -   利用在大規模資料集上訓練的和弦 N-gram 模型計算該序列的 Perplexity (或 Negative Log Likelihood)。
            -   **高 CPI** 代表和弦進行不尋常或缺乏規律。
        4.  **Structureness Indicator (SI)**：
            -   計算音訊或 MIDI 的 Self-Similarity Matrix (SSM)。
            -   計算 **Fitness Scape Plot**，透過在 SSM 對角線上整合不同時間尺度的重複性，生成多尺度的結構圖。
            -   **高 SI** 代表曲子具有明顯的重複結構（如 Verse-Chorus 形式）。
    -   **潛在風險**：
        -   **指標與聽感脫節**：客觀指標無法完全反映主觀好壞（例如熵過低=單調）。
        -   **計算成本**：Scape Plot 計算耗時。

## 7. 條件傳遞與架構重構 (Conditioning & Recursion)
-   **問題**：
    -   `fractalgen` 參考了空間相鄰資訊，但音樂生成需要考慮「時間前後 (Temporal Context)」與「音高高低 (Pitch Context)」，且不能僅依賴簡單的線性插值上採樣，需捕捉非相鄰的音樂結構。
    -   目前的 `ModuleList` 迭代架構不夠靈活，難以動態調整層級，且偏離了 Fractal 自相似的遞歸精神。
-   **改進方向**：
    -   **演算法改進 (Hierarchical Context Aggregation)**：
        1.  **上下文視窗 (Temporal Context)**：在 Level $L$ 接收 Level $L-1$ 的條件時，不只取對應時間點 $t$，而是取一個視窗 $[t-k, t+k]$ 的 Embedding 進行聚合（例如透過 Conv1d 或 Attention），明確提供「過去」與「未來」的宏觀結構資訊。
        2.  **音高與和聲上下文 (Pitch/Harmonic Context)**：透過 Global Condition (Chroma) 投影，並讓模型在每一層都能 Cross-Attend 到 Global Chroma，確保生成的音高符合全局調性。
        3.  **避免單純相鄰**：引入 **Cross-Attention to Parent** 機制。讓 Level $L$ 的 Token 可以 Query Level $L-1$ 的所有 Token（或大範圍 Window），從而捕捉重複樂句或呼應結構，而非僅依賴對齊的父節點。
    -   **架構重構 (Recursive Refactoring)**：
        -   將 `TemporalFractalNetwork` 重構為遞歸類別 `FractalGen`。
        -   每個 `FractalGen` 實例包含一個 `generator` 和一個 `next_fractal` (或者是 `TokenOutput`)。
        -   這樣能更自然地處理任意深度的 Fractal 結構。

## 8. 視覺化與驗證流程 (Visualization & Validation)
-   **問題**：
    -   目前的 Visualization 流程在處理多層級、變長序列時容易發生 Shape Mismatch 錯誤。
    -   GIF 生成 (`create_growth_animation`) 忽略了不同層級的時間解析度差異，導致動畫崩潰或不準確。
    -   缺乏穩定的 WebUI 介面供使用者互動。
-   **改進方向**：
    -   **視覺化修正**：在 `trainer.py` 中，於 logging 前強制執行 **Upsampling (Interpolation)**，將所有層級的輸出統一到最大解析度 (Max Length)，確保 Tensorboard 和 GIF 生成時維度一致。
    -   **驗證增強**：在 `validation_step` 中加入 `try-except` 區塊與維度檢查，若生成失敗則跳過該 Batch 並記錄錯誤，避免訓練中斷。

## 9. 現有 WebUI 的改進 (Existing WebUI Improvements)
-   **問題**：
    -   **條件控制不足**：現有 API 缺乏對 **Global Chroma Condition** 的直接支援，且無法針對特定 Fractal 層級（如只重生成結構 Level 0）進行操作。
    -   **互動性弱**：WebSocket 串流僅用於進度條，缺乏「畫布即時重繪 (Interactive Inpainting)」能力。
    -   **視覺化單一**：生成的 PNG/GIF 未能分層展示（Structure vs. Content），難以觀察 Fractal 生成的內部邏輯。
    -   **資源管理**：`outputs` 資料夾缺乏自動清理機制，長期運行會累積大量暫存檔。
-   **改進方向**：
    -   **增強 API**：更新 `GenerationRequest` 以支援 `global_chroma` 輸入（可從上傳的 Audio/MIDI 提取）。新增 `target_levels` 參數，允許僅對特定層級進行採樣。
    -   **分層視覺化**：前端介面應新增「層級檢視器 (Layer Inspector)」，允許使用者切換查看 Level 0 (Density/Tempo 曲線) 與 Level 1+ (Piano Roll) 的疊加視圖。
    -   **互動模式**：實作基於 WebSocket 的 **Interactive Canvas**，當使用者在前端修改 Note 時，後端僅對受影響的局部區域與相關層級進行快速 Re-sampling。
    -   **任務管理**：實作 `JobManager` 的定期清理功能，自動刪除超過時限（如 24 小時）的舊 Job 輸出。

## Implementation Notes & Risks (2025-12-07)

### Completed Refactoring
- **Recursive Architecture**: Refactored `TemporalFractalNetwork` to `RecursiveFractalNetwork` (using `FractalGen` nodes).
- **Decoder Upgrade**: Replaced LSTM decoder with Transformer Decoder (2 layers, 1024 dim) + KV Cache.
- **Velocity Classification**: Changed velocity generation to 32-bin classification.
- **Relative Attention**: Integrated Music Transformer style Relative Global Attention.
- **Dynamic Loss Weighting**: Implemented learnable `sigma` parameters for multi-task loss balancing.
- **Metrics**: Added Pitch Class Entropy, Groove Similarity, and Structureness Indicator.

### Potential Risks & Issues
1.  **Checkpoint Compatibility**: The new `RecursiveFractalNetwork` structure is **not compatible** with old `TemporalFractalNetwork` checkpoints. The state dict keys have changed from `levels.0...` to `root.layer...` and `root.next_level...`. Fine-tuning old models requires a migration script.
2.  **Memory usage with Relative Attention**: `RelativeGlobalAttention` computes an $(L, L)$ matrix. While acceptable for current sequence lengths (up to 512), scaling to very long sequences might require sparse attention mechanisms (e.g., Museformer) as noted in Issue 5.
3.  **Training Stability with New Loss**: The introduction of learnable loss weights ($\sigma$) and the switch to Cross Entropy for velocity (magnitude ~2-4) vs previous MSE (magnitude ~0.1) significantly alters the loss landscape. Learning rates might need retuning.
4.  **Velocity Quantization**: 32-bin quantization is a heuristic. If velocity resolution is felt to be too low, consider increasing to 64 or 128 bins.
5.  **WebUI Integration**: The WebUI code in `src/web` has not yet been updated to use the new `RecursiveFractalNetwork` class. Inference via WebUI will likely fail until updated.

