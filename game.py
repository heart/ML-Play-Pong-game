import os
import time
import math
import random

sceneSize = 24

# ✅ Random Ball Position & Speed
def reset_ball():
    ballX = random.randint(2, sceneSize - 3)  # เลี่ยงขอบ
    ballY = random.randint(2, sceneSize // 2)  # ไม่เริ่มใกล้ Paddle เกินไป
    ballXSpeed = random.choice([-1, 1])  # สุ่มทิศทางซ้าย/ขวา
    ballYSpeed = random.choice([-1, 1])  # สุ่มทิศทางขึ้น/ลง
    return ballX, ballY, ballXSpeed, ballYSpeed

# ✅ กำหนดค่าเริ่มต้นของเกมแบบสุ่ม
ballX, ballY, ballXSpeed, ballYSpeed = reset_ball()

paddleWidth = 3
paddleX = math.floor(sceneSize / 2)
paddleY = sceneSize - 1 

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def tick():
    global ballX, ballY, ballXSpeed, ballYSpeed, paddleX, paddleY
    
    gameUpdate()

    # ✅ เช็คว่าบอลโดน Paddle หรือเปล่า
    hit = False
    game_over = False

    if ballY == paddleY - 1 and paddleX <= ballX < paddleX + paddleWidth:
        ballYSpeed *= -1  # เด้งกลับ
        hit = True  # ✅ AI ควรได้รับ Reward

    # ✅ เช็คว่าบอลตกขอบล่าง (Game Over)
    if ballY >= sceneSize - 1:
        game_over = True

    return hit, game_over


def gameUpdate():
    global ballX, ballY, ballXSpeed, ballYSpeed
    ballX += ballXSpeed
    ballY += ballYSpeed

    # ✅ ถ้าบอลชนขอบซ้ายหรือขวา → เด้งกลับ
    if ballX == 0 or ballX == sceneSize - 1:
        ballXSpeed *= -1

    # ✅ ถ้าบอลชนขอบบน → เด้งกลับ
    if ballY == 0:
        ballYSpeed *= -1

def render():
    clear_screen()
    for i in range(sceneSize):
        for j in range(sceneSize):
            if i == ballY and j == ballX:
                print("O", end="")
            elif i == paddleY and paddleX <= j < paddleX + paddleWidth:
                print("=", end="")
            else:
                print(" ", end="")
        print("")

def reset():
    """ ✅ รีเซ็ตเกมโดยสุ่มตำแหน่งลูกบอลใหม่ """
    global ballX, ballY, ballXSpeed, ballYSpeed, paddleX
    ballX, ballY, ballXSpeed, ballYSpeed = reset_ball()
    paddleX = math.floor(sceneSize / 2)  # Paddle กลับมาตรงกลาง

def move_left():
    global paddleX
    if paddleX > 0:
        paddleX -= 1

def move_right():
    global paddleX
    if paddleX + paddleWidth < sceneSize:
        paddleX += 1

def get_state():
    """ ✅ ส่งสถานะของเกม """
    state = [[0] * sceneSize for _ in range(sceneSize)]
    state[ballY][ballX] = 1  # Mark Ball
    for j in range(paddleX, paddleX + paddleWidth):
        state[paddleY][j] = 2  # Mark Paddle

    return state, ballXSpeed, ballYSpeed
