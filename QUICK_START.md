# FractalMIDI 快速啟動指南

## 前置準備

### 1. 準備 Checkpoint 檔案

將訓練好的模型 checkpoint 放到指定目錄：

```bash
mkdir -p outputs/checkpoints
# 將 .ckpt 檔案複製到這個目錄
cp /path/to/your/checkpoint.ckpt outputs/checkpoints/
```

**注意**：應用程式會自動載入最新的 checkpoint（根據 step 數字排序）

### 2. 準備驗證集範例（選用）

如果要使用 conditional 或 inpainting 模式，需要準備 MIDI 範例：

```bash
mkdir -p dataset/validation_examples
# 將 MIDI 檔案複製到這個目錄
cp /path/to/midi/files/*.mid dataset/validation_examples/
```

應用程式會自動掃描並建立縮圖。

## 啟動方式

### 方式 1：直接運行（開發模式）

```bash
# 1. 安裝依賴
pip install -r requirements.txt
pip install -r web/requirements_web.txt

# 2. 啟動伺服器
cd web/backend
python app.py

# 3. 開啟瀏覽器
# 訪問 http://localhost:8000/static/index.html
```

### 方式 2：使用 Uvicorn（生產模式）

```bash
cd web/backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 方式 3：使用 Docker

```bash
# 1. 確保 checkpoint 在正確位置
ls outputs/checkpoints/

# 2. 啟動容器
docker-compose up -d

# 3. 查看日誌
docker-compose logs -f fractalmidi

# 4. 開啟瀏覽器
# 訪問 http://localhost:8000/static/index.html
```

## 自動載入功能

應用程式啟動時會：

1. **自動掃描** `outputs/checkpoints/` 目錄中的所有 `.ckpt` 檔案
2. **自動載入** 最新的 checkpoint（根據檔名中的 step 數字）
3. **自動掃描** `dataset/validation_examples/` 中的 MIDI 檔案（如果存在）
4. **自動建立** 縮圖和 metadata

### 查看載入狀態

啟動後查看日誌：

```bash
# 直接運行時
# 在終端機中會看到：
# INFO - Auto-loaded latest checkpoint: step=10000
# INFO - Found 5 checkpoints
# INFO - Loaded 20 examples from metadata

# Docker 運行時
docker-compose logs fractalmidi | grep -E "checkpoint|examples"
```

## 使用介面

### 1. 模型配置

- **Generator Type**: 自動選擇 MAR（推薦）
- **Scan Order**: AR 模式時可選擇 row_major 或 column_major
- **Checkpoint**: 下拉選單會顯示所有可用的 checkpoint，預設選擇最新的

### 2. 生成模式

#### Unconditional（無條件生成）
- 從零開始生成音樂
- 調整 Length 參數控制生成長度

#### Conditional（條件生成）
- 從範例選單選擇一個 MIDI 檔案
- 設定 Condition Length（前綴長度）
- 設定 Total Length（總長度）
- 模型會延續前綴的風格

#### Inpainting（修補生成）
- 從範例選單選擇一個 MIDI 檔案
- 使用 Eraser Tool 在 canvas 上標記要重新生成的區域
- 模型會填補被標記的區域

### 3. 進階參數

- **Temperature**: 控制隨機性（0.5-2.0）
  - 低溫度 = 更保守、更可預測
  - 高溫度 = 更隨機、更有創意
  
- **CFG Scale**: Classifier-Free Guidance（1.0-3.0）
  - 1.0 = 無引導
  - 更高 = 更強的條件引導

- **Visualization**:
  - Create GIF: 生成動畫 GIF
  - Show Progress: 在 GIF 中顯示進度條
  - Show Grid: 在 GIF 中顯示網格

### 4. 生成流程

1. 選擇模型和參數
2. 點擊 "✨ Generate" 按鈕
3. 觀看即時進度更新
4. 查看生成結果
5. 下載 MIDI、圖片或 GIF

## 常見問題

### Q: 啟動時顯示 "No checkpoints found"

**A**: 確認 checkpoint 檔案在正確位置：

```bash
ls -lh outputs/checkpoints/
# 應該看到 .ckpt 檔案
```

### Q: 無法連接到伺服器

**A**: 檢查：
1. 伺服器是否正在運行
2. 防火牆設定
3. 端口 8000 是否被佔用

```bash
# 檢查端口
lsof -i :8000

# 更換端口（在 config.py 中修改）
```

### Q: 生成很慢

**A**: 
1. 確認使用 GPU（檢查 `CUDA_VISIBLE_DEVICES`）
2. 減少生成長度
3. 降低 num_iter_list 參數

### Q: WebSocket 連接失敗

**A**: 
1. 確認瀏覽器支援 WebSocket
2. 檢查 CORS 設定
3. 嘗試使用 REST API 模式（POST /api/generate）

### Q: 找不到驗證集範例

**A**: 
1. 確認 MIDI 檔案在 `dataset/validation_examples/`
2. 重新啟動伺服器讓它掃描檔案
3. 檢查日誌中的錯誤訊息

## 測試 AR 修復

在使用 web 應用程式之前，建議先測試 AR 修復：

```bash
# 快速測試（5分鐘）
python test_ar_fixes.py --quick

# 完整測試（30-60分鐘）
python test_ar_fixes.py --full
```

## 效能建議

### GPU 記憶體不足

如果遇到 CUDA out of memory：

1. 減少 batch_size（在 config.py 中）
2. 減少生成長度
3. 使用較小的模型

### CPU 模式

如果沒有 GPU，修改 `web/backend/config.py`：

```python
@dataclass
class ModelConfig:
    device: str = "cpu"  # 改為 "cpu"
```

**注意**：CPU 模式會非常慢！

## 監控和日誌

### 查看日誌

```bash
# 直接運行
# 日誌會輸出到終端機

# Docker 運行
docker-compose logs -f fractalmidi

# 日誌檔案
tail -f logs/fractal_midi_web.log
```

### 監控生成任務

使用 API 查詢狀態：

```bash
# 查看系統狀態
curl http://localhost:8000/api/status

# 查看模型資訊
curl http://localhost:8000/api/models/info

# 查看任務狀態
curl http://localhost:8000/api/generate/{job_id}
```

## 下一步

1. ✅ 啟動應用程式
2. ✅ 自動載入 checkpoint
3. ✅ 測試無條件生成
4. ✅ 測試條件生成（需要範例）
5. ✅ 測試 inpainting（需要範例）
6. ✅ 調整參數找到最佳設定
7. ✅ 整合到 TouchDesigner（參考 `TOUCHDESIGNER_INTEGRATION.md`）

## 技術支援

遇到問題時：

1. 查看日誌檔案
2. 檢查 `WEB_APPLICATION_README.md`
3. 參考 `IMPLEMENTATION_COMPLETE.md`
4. 查看 API 文檔：http://localhost:8000/docs

---

**祝您使用愉快！🎵✨**

