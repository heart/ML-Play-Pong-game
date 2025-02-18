import os
import time
import random
import math
import sys

class PongGame:
    def __init__(self, width=30, height=20, paddle_size=7):
        self.width = width
        self.height = height
        self.paddle_size = paddle_size
        
        # Initialize game state
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        # Ball position and speed
       
        self.ball_x = random.randint(0, self.width-1)
        self.ball_y = random.randint(0, math.floor(self.height/2)-1)
        self.ball_speed_x = random.choice([-1, 1])
        self.ball_speed_y = random.choice([-1, 1])
        
        # Paddle position (paddle จะอยู่ด้านล่าง แนวนอน)
        self.paddle_x = self.width // 2
        self.paddle_y = self.height - 1
        
        # Score
        self.hits = 0
        return self.get_state()

    def get_state(self):
        """Return current game state"""
        return {
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'ball_speed_x': self.ball_speed_x,
            'ball_speed_y': self.ball_speed_y,
            'paddle_x': self.paddle_x,
            'paddle_y': self.paddle_y
        }

    def moveLeft(self):
        """Move paddle left (ลดค่า x)"""
        if self.paddle_x > self.paddle_size // 2:
            self.paddle_x -= 1

    def moveRight(self):
        """Move paddle right (เพิ่มค่า x)"""
        if self.paddle_x < self.width - (self.paddle_size // 2) - 1:
            self.paddle_x += 1

    def tick(self):
        """Progress game by one frame"""
        # Update ball position
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # ตรวจสอบขอบในแนวแกน x
        if self.ball_x < 0:
            self.ball_x = 0
            self.ball_speed_x *= -1
        elif self.ball_x >= self.width:
            self.ball_x = self.width - 1
            self.ball_speed_x *= -1

        # ตรวจสอบขอบบนในแนวแกน y
        if self.ball_y < 0:
            self.ball_y = 0
            self.ball_speed_y *= -1
        # ไม่ต้องตีกลับด้านล่างเพราะ paddle อยู่ด้านล่าง

        # ตรวจสอบการชนกับ paddle (ที่ด้านล่าง)
        hit = False
        if self.ball_y >= self.height - 2:
            paddle_left = self.paddle_x - self.paddle_size // 2
            paddle_right = self.paddle_x + self.paddle_size // 2

            nextX = self.ball_x + self.ball_speed_x
            nextY = self.ball_y + self.ball_speed_y
            if nextX >= paddle_left and nextX <= paddle_right and nextY == self.paddle_y:
                self.ball_speed_y *= -1
                self.hits += 1
                hit = True

        # ตรวจสอบว่าลูกบอลหลุดออกจาก paddle แล้วหรือยัง
        done = self.ball_y >= self.height - 1

        return hit, done

    def render(self):
        """Render game state to terminal ด้วยการ concat string แล้ว print ครั้งเดียว"""
        # ย้ายเคอร์เซอร์ไปที่ตำแหน่ง home และล้างหน้าจอทั้งหมด
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()
        
        # สร้างรายการสำหรับเก็บแต่ละบรรทัด
        lines = []
        
        # วาดขอบบนของเกม
        border = '-' * (self.width + 2)
        lines.append(border)
        
        # สร้าง board ว่าง
        board = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # วาดลูกบอล
        if 0 <= self.ball_y < self.height and 0 <= self.ball_x < self.width:
            board[self.ball_y][self.ball_x] = 'O'
        
        # วาด paddle แนวนอนที่ด้านล่าง
        for i in range(self.paddle_size):
            x = self.paddle_x - (self.paddle_size // 2) + i
            if 0 <= x < self.width:
                board[self.paddle_y][x] = '='
        
        # สร้างบรรทัดของ board
        for row in board:
            lines.append('|' + ''.join(row) + '|')
        
        # วาดขอบล่างของเกม
        lines.append(border)
        
        # แสดงคะแนน
        lines.append(f'Hits: {self.hits}')
        
        # รวมทุกบรรทัดเป็น string เดียวแล้ว print ครั้งเดียว
        output = "\n".join(lines)
        print(output)


# def play_manual():
#     """Function to play the game manually for testing"""
#     game = PongGame(30, 20, paddle_size=7)
#     done = False
    
#     while not done:
#         game.render()
        
#         # รับอินพุต (non-blocking)
#         import sys
#         import select
        
#         if sys.platform != 'win32':  # สำหรับ Unix-like systems
#             rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
#             if rlist:
#                 key = sys.stdin.read(1)
#                 if key == 'a':
#                     game.moveLeft()
#                 elif key == 'd':
#                     game.moveRight()
#         else:  # สำหรับ Windows
#             import msvcrt
#             if msvcrt.kbhit():
#                 key = msvcrt.getch().decode('utf-8')
#                 if key == 'a':
#                     game.moveLeft()
#                 elif key == 'd':
#                     game.moveRight()
        
#         hit, done = game.tick()
#         time.sleep(0.1)  # ควบคุมความเร็วของเกม
        
#     game.render()
#     print("Game Over!")
#     print(f"Final Score: {game.hits}")

# if __name__ == "__main__":
#     play_manual()
