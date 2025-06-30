# ADB-sudoku-solver

利用 ADB 自動操作模擬器，結合 CNN 圖像辨識與數獨求解器，自動玩遊戲並將結果即時回傳至 Discord！

## 簡介

結合電腦視覺、深度學習、遊戲自動化與 Discord bot 的完整系統

- 從 Android 模擬器擷取遊戲畫面
- 使用 CNN 模型辨識數獨盤的每個數字
- 自動解出數獨並透過 ADB 操作點擊填入
- 將擷取結果、解題時間與遊戲狀態回報至 Discord 頻道

---

## 使用技術

| 技術 | 用途 |
|------|------|
| Python | 程式語言主體 |
| PyTorch | 手寫數字辨識 CNN 模型 |
| Pillow / OpenCV | 畫面裁切與處理 |
| ADB | 控制模擬器操作 |
| discord.py | Discord bot 通訊 |
| asyncio | 非同步邏輯執行 |

---

## 🖼系統流程

```mermaid
graph TD
    A[ADB 擷取畫面] --> B[裁切數獨區域]
    B --> C[9x9 分格]
    C --> D[CNN 模型 OCR]
    D --> E[數獨解題器 DFS]
    E --> F[ADB 自動填入]
    E --> G[回傳 Discord 訊息]
