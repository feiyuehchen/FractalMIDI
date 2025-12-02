# TouchDesigner 快速入門指南

## 最簡單的開始方式

### 步驟 1：啟動 FractalMIDI 伺服器

```bash
cd /home/feiyueh/FractalMIDI/web/backend
python app.py
```

伺服器會在 `http://localhost:8000` 啟動

### 步驟 2：在 TouchDesigner 中建立 WebSocket 連接

1. **新增 Web Client DAT**
   - 在 TouchDesigner 中按 `Tab` 鍵
   - 輸入 "Web Client"
   - 選擇 `Web Client DAT`

2. **設定 WebSocket 連接**
   - 在 Web Client DAT 的參數面板中：
     - Active: ✓ (勾選)
     - Request: WebSocket
     - WebSocket URL: `ws://localhost:8000/ws/generate`
     - Auto Reconnect: ✓ (勾選)

### 步驟 3：發送生成請求

建立一個 **Text DAT**，命名為 `generate_request`，內容：

```json
{
  "mode": "unconditional",
  "generator_type": "mar",
  "length": 256,
  "temperature": 1.0,
  "cfg": 1.0,
  "create_gif": false,
  "show_progress": false
}
```

建立一個 **Button COMP**，在其 callbacks 中加入：

```python
def onOffToOn(comp, prev):
    # 發送生成請求
    request_text = op('generate_request').text
    op('webclient1').sendText(request_text)
```

### 步驟 4：接收和顯示結果

在 Web Client DAT 的 callbacks 中加入：

```python
def onReceiveText(dat, text):
    import json
    
    # 解析回應
    data = json.loads(text)
    
    # 更新進度
    if 'progress' in data:
        progress = data['progress']
        op('progress_bar').par.value0 = progress
        print(f"Progress: {progress*100:.1f}%")
    
    # 處理完成
    if data.get('status') == 'completed':
        print("Generation completed!")
        
        # 取得圖片 URL
        image_url = data.get('image_url')
        if image_url:
            # 下載並顯示圖片
            full_url = f"http://localhost:8000{image_url}"
            op('moviefilein1').par.file = full_url
            
        # 取得 MIDI URL
        midi_url = data.get('midi_url')
        if midi_url:
            print(f"MIDI available at: http://localhost:8000{midi_url}")
```

## 完整的 TouchDesigner 網路範例

```
[Button COMP]
    ↓ (onOffToOn callback)
[Text DAT: generate_request]
    ↓
[Web Client DAT: webclient1]
    ↓ (onReceiveText callback)
[Movie File In TOP: moviefilein1] → 顯示生成的 piano roll
[Slider COMP: progress_bar] → 顯示進度
```

## 進階功能

### 1. 即時視覺化

建立一個 **GLSL TOP** 來渲染 piano roll：

```glsl
// 在 GLSL TOP 的 pixel shader 中
uniform sampler2D sPianoRoll;
uniform float uProgress;

out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;
    
    // 讀取 piano roll
    vec4 color = texture(sPianoRoll, uv);
    
    // 添加生長動畫
    float dist = length(uv - vec2(0.5, 0.5));
    float growth = smoothstep(0.0, 1.0, uProgress - dist);
    
    color.a *= growth;
    
    // 添加發光效果
    if (color.r > 0.1) {
        color.rgb += vec3(0.3, 0.2, 0.1) * (1.0 - growth);
    }
    
    fragColor = color;
}
```

### 2. 互動式 Inpainting

使用 **Panel COMP** 建立觸控介面：

```python
# 在 Panel COMP 的 panel callbacks 中

def onValueChange(comp, rows, cols, prev):
    # 偵測觸控位置
    if len(rows) > 0 and len(cols) > 0:
        x = cols[0].val
        y = rows[0].val
        
        # 轉換為 piano roll 座標
        time_start = int(x * 256)
        time_end = time_start + 32
        
        # 建立 inpainting 請求
        request = {
            "mode": "inpainting",
            "inpaint_example_id": "current_state",
            "inpaint_mask": [[time_start, time_end]],
            "length": 256,
            "generator_type": "mar",
            "temperature": 1.0
        }
        
        # 發送請求
        import json
        op('webclient1').sendText(json.dumps(request))
```

### 3. 粒子系統（音符彈出效果）

使用 **Particle GPU TOP** 建立音符彈出動畫：

1. **建立 Particle GPU**
   - 設定 particle 數量：1000
   - Life: 2 秒
   - Speed: 隨機

2. **連接到 MIDI 資料**
   - 當新音符出現時，發射粒子
   - 粒子顏色對應音符速度
   - 粒子位置對應音高和時間

3. **視覺效果**
   - 使用 **Blur TOP** 添加模糊
   - 使用 **Composite TOP** 疊加到 piano roll 上
   - 使用 **Feedback TOP** 建立拖尾效果

## 效能優化

### 減少延遲
```python
# 在 Web Client DAT 中
# 使用較短的生成長度
request = {
    "length": 128,  # 而不是 256
    "create_gif": False  # 不生成 GIF 以加快速度
}
```

### GPU 加速
- 使用 GPU-based TOPs（Particle GPU, GLSL TOP）
- 避免使用 CPU-based operations
- 限制粒子數量

### 記憶體管理
```python
# 定期清理舊的生成結果
def clearOldGenerations():
    # 只保留最近 5 個結果
    # 刪除舊的圖片檔案
    pass
```

## 範例專案結構

```
TouchDesigner Project/
├── fractalmidi_simple.toe          # 簡單範例
├── fractalmidi_interactive.toe     # 互動範例
├── fractalmidi_performance.toe     # 表演用範例
└── components/
    ├── websocket_client.tox        # WebSocket 元件
    ├── pianoroll_viz.tox           # Piano roll 視覺化
    ├── particle_system.tox         # 粒子系統
    └── touch_interface.tox         # 觸控介面
```

## 常見問題

### Q: WebSocket 連接失敗
**A**: 
1. 確認 FractalMIDI 伺服器正在運行
2. 檢查 URL 是否正確：`ws://localhost:8000/ws/generate`
3. 查看 TouchDesigner 的 textport 錯誤訊息

### Q: 圖片無法載入
**A**:
1. 確認 URL 格式：`http://localhost:8000/outputs/{job_id}/output.png`
2. 使用 **Download TOP** 而不是直接 Movie File In
3. 檢查檔案權限

### Q: 生成太慢
**A**:
1. 減少 length 參數
2. 使用 MAR 而不是 AR
3. 確認伺服器使用 GPU

## 測試連接

在 TouchDesigner 的 **Textport** 中測試：

```python
# 測試 WebSocket 連接
import websocket
import json

ws = websocket.create_connection("ws://localhost:8000/ws/generate")

request = {
    "mode": "unconditional",
    "length": 128,
    "temperature": 1.0
}

ws.send(json.dumps(request))

# 接收回應
while True:
    result = ws.recv()
    data = json.loads(result)
    print(data)
    if data.get('status') == 'completed':
        break

ws.close()
```

## 下一步

1. ✅ 建立基本 WebSocket 連接
2. ✅ 測試生成請求
3. ✅ 顯示生成結果
4. ✅ 添加視覺效果
5. ✅ 建立互動介面
6. ✅ 優化效能
7. ✅ 準備表演/展覽

## 完整文件

詳細的整合指南請參考：
- **`web/TOUCHDESIGNER_INTEGRATION.md`** - 完整技術文件
- **`WEB_APPLICATION_README.md`** - API 文件
- **`QUICK_START.md`** - 快速啟動指南

---

**開始創作吧！🎵✨**

