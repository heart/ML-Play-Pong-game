import numpy as np
import tensorflow as tf
from collections import deque
from game import PongGame  # ตรวจสอบให้แน่ใจว่ามีการ implement คลาส PongGame ไว้แล้ว

# Agent ที่ใช้เล่นเกม (โครงสร้างควรตรงกับที่ใช้เทรน)
class PongDQNAgent:
    def __init__(self, state_size=15, action_size=3):
        self.state_size = state_size  
        self.action_size = action_size  
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        return model
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

# ฟังก์ชันสำหรับประมวลผล state และจัดการ history ของ frame (3 เฟรม)
def process_state(game_state, history_buffer):
    current_state = np.array([
        game_state['ball_x'],
        game_state['ball_y'],
        game_state['ball_speed_x'],
        game_state['ball_speed_y'],
        game_state['paddle_x']
    ])
    
    history_buffer.append(current_state)
    if len(history_buffer) > 3:
        history_buffer.popleft()
    
    return np.concatenate(list(history_buffer))

# ฟังก์ชันสำหรับเล่นเกมโดยใช้ agent ที่เทรนแล้ว
def play_game(agent, game):
    # เตรียม history buffer (3 เฟรม)
    history_buffer = deque(maxlen=3)
    for _ in range(3):
        history_buffer.append(np.zeros(5))
    
    done = False
    while not done:
        # รับ state ปัจจุบัน (รวมประวัติ 3 เฟรม)
        state = process_state(game.get_state(), history_buffer)
        
        # เลือก action โดยใช้โมเดลที่เทรนแล้ว
        action = agent.act(state)
        
        # ดำเนินการ action:
        # action 0: moveLeft, action 1: stay, action 2: moveRight
        if action == 0:
            game.moveLeft()
        elif action == 2:
            game.moveRight()
        # action 1 ไม่ต้องทำอะไร (stay)
        
        # ให้เกมก้าวไปข้างหน้า
        hit, done = game.tick()
        game.render()

if __name__ == "__main__":
    # สร้างอินสแตนซ์ของเกม
    game = PongGame(24, 24, 4)
    
    # สร้าง agent และโหลดโมเดลที่เทรนเสร็จแล้ว
    agent = PongDQNAgent()
    agent.model.load_weights("model_final.h5")
    print("โหลดน้ำหนักโมเดลจาก model_final.h5 เรียบร้อยแล้ว")
    
    # เล่นเกมด้วยโมเดลที่เทรนแล้ว
    play_game(agent, game)
