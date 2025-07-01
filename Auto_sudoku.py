import asyncio
import subprocess
import time
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import discord
import asyncio

TOKEN = "Your Token"
CHANNEL_ID = id
ADB_DEVICE = "127.0.0.1:port"

log_channel = None
intents = discord.Intents.default()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    global log_channel
    log_channel = client.get_channel(CHANNEL_ID)
    print(f"Bot 已上線：{client.user}")
    await main_loop()

async def send_log(content):
    print(content)
    if log_channel:
        try:
            await log_channel.send(f"{content}")
        except Exception as e:
            print(f"發送 Discord 訊息失敗：{e}")

async def main_loop():
    while True:
        await asyncio.sleep(2)
        adb_run(["shell", "input", "tap", str(360), str(1192)])
        await asyncio.sleep(1)
        capture_full_screenshot()
        crop_sudoku_board()
        cells = split_board_to_cells()
        original_board = get_board_from_cells(cells)

        board_text = "\n".join(str(row) for row in original_board)
        await send_log(f"# 擷取結果：\n```\n{board_text}\n```")
        
        start_time = time.time()
        solved_board = [row.copy() for row in original_board]

        # 轉為非同步執行，避免卡住 event loop
        solved = await asyncio.to_thread(solve, solved_board)
        
        if solve(solved_board):
            duration = round(time.time() - start_time, 2)
            await send_log(f"解題成功，耗時 {duration} 秒，開始填入答案...")
            fill_all(solved_board, original_board)
            await send_log("填入成功，即將開啟新遊戲，難度: 6/6")
            new_game()
        else:
            await send_log("無解，即將開啟新遊戲，難度: 6/6")
            adb_run(["shell", "input", "tap", str(34), str(40)])
            await asyncio.sleep(1)
            adb_run(["shell", "input", "tap", str(360), str(1135)])
            await asyncio.sleep(1)
            adb_run(["shell", "input", "tap", str(360), str(1105)])

        adb_run(["shell", "input", "tap", str(612), str(852)])
        await asyncio.sleep(0.1)
        adb_run(["shell", "input", "tap", str(612), str(852)])

x_map = {1: 128, 2: 181, 3: 243, 4: 302, 5: 363, 6: 421, 7: 477, 8: 535, 9: 592}
y_map = {1: 810, 2: 748, 3: 692, 4: 633, 5: 577, 6: 520, 7: 459, 8: 400, 9: 342}

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = DigitCNN()
model.load_state_dict(torch.load('digit_cnn.pt', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def adb_run(cmd_list):
    full_cmd = ["adb", "-s", ADB_DEVICE] + cmd_list
    subprocess.run(full_cmd)
    
def tap_number(number):
    positions = {
        1: (110, 1114),
        2: (171, 1114),
        3: (232, 1114),
        4: (293, 1114),
        5: (354, 1114),
        6: (415, 1114),
        7: (476, 1114),
        8: (537, 1114),
        9: (601, 1114),
    }
    if number in positions:
        x, y = positions[number]
        adb_run(["shell", "input", "tap", str(x), str(y)])
    else:
        print(f"無效數字：{number}")

def tap_board(x, y):
    if x in x_map and y in y_map:
        xpos, ypos = x_map[x], y_map[y]
        adb_run(["shell", "input", "tap", str(xpos), str(ypos)])
    else:
        print(f"格子座標無效: x={x}, y={y}")

def capture_full_screenshot(filename="full.png"):
    adb_run(["shell", "screencap", "-p", "/sdcard/full.png"])
    adb_run(["pull", "/sdcard/full.png", filename])
    print(f"擷取畫面完成: {filename}")

def crop_sudoku_board(src_file="full.png", out_file="board.png"):
    img = Image.open(src_file)
    cropped = img.crop((98, 316, 621, 839))  # (left, top, right, bottom)
    cropped.save(out_file)
    print(f"裁切數獨盤存為: {out_file}")
    return cropped

def split_board_to_cells(image_path="board.png"):
    img = Image.open(image_path)
    cell_width = img.width // 9
    cell_height = img.height // 9

    cells = []
    for row in range(9):
        row_cells = []
        for col in range(9):
            left = col * cell_width + 5
            top = row * cell_height + 5
            right = (col + 1) * cell_width - 5
            bottom = (row + 1) * cell_height - 5
            cell_img = img.crop((left, top, right, bottom))
            row_cells.append(cell_img)
        cells.append(row_cells)
    print("分割並裁切為 9x9")
    return cells

# 預先載入模板（只載一次以提升效能）
def load_templates(template_dir='D:/JupyterServer/Auto Sudoku/image'):
    templates = {}
    for i in range(10):
        path = os.path.join(template_dir, f"{i}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[i] = img
    return templates

templates = load_templates()

def ocr_cell(cell_img):
    """
    cell_img: PIL.Image 格式的單格圖片
    返回: 預測的數字 0~9 (0代表空白)
    """
    # 轉成模型輸入格式
    input_tensor = transform(cell_img).unsqueeze(0)  # (1, 1, 48, 48)

    # 模型推論
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # 可加閾值判斷，低信心回傳0（空白）
    if conf.item() < 0.5:
        return 0
    else:
        return pred.item()

def get_board_from_cells(cells):
    board = []
    for row in cells:
        board_row = [ocr_cell(cell) for cell in row]
        board.append(board_row)
    return board

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[row//3*3 + i//3][col//3*3 + i%3] == num:
            return False
    return True

def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def fill_cell(x, y, number):
    tap_board(x, y)
    time.sleep(0.2)
    tap_number(number)
    time.sleep(0.2)

def fill_all(board, original):
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0:  # 只有空格才填
                fill_cell(j + 1, 9 - i, board[i][j])

def new_game():
    time.sleep(1)
    adb_run(["shell", "input", "tap", str(33), str(42)])
    time.sleep(1)
    adb_run(["shell", "input", "tap", str(100), str(1000)])
    time.sleep(1)
    adb_run(["shell", "input", "tap", str(100), str(1000)])
    time.sleep(1)
    adb_run(["shell", "input", "tap", str(360), str(1154)])
    time.sleep(1)
    adb_run(["shell", "input", "tap", str(360), str(1105)])
    time.sleep(1)

client.run(TOKEN)