import game
import tensorflow as tf
import numpy as np
import random
import time
import os
from collections import deque
import matplotlib.pyplot as plt

# ปิด Log ของ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ตรวจสอบว่าใช้ GPU ได้ไหม
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ ใช้ GPU สำหรับ Training!")
else:
    print("⚠️ ไม่พบ GPU, ใช้ CPU แทน")

# พารามิเตอร์ของ RL
gamma = 0.95  # Discount Factor
epsilon = 1.0  # ค่าเริ่มต้นของ Exploration
epsilon_min = 0.05  # ค่า Epsilon ต่ำสุด
epsilon_decay = lambda ep: max(epsilon_min, epsilon * np.exp(-0.001 * ep))  # ปรับการลด epsilon
tau = 0.01  # Soft update ระหว่าง Model หลักและ Target Model
learning_rate = 0.001
batch_size = 32
max_memory_size = 10000
memory = deque(maxlen=max_memory_size)
scores = []
num_frames = 4

state_size = (game.sceneSize, game.sceneSize, num_frames * 2)  # *2 เพราะมี speed channel
action_size = 3  # 3 Actions: ซ้าย, อยู่กับที่, ขวา

# ฟังก์ชันสร้าง Dueling DQN Model
def build_dueling_dqn():
    input_layer = tf.keras.layers.Input(shape=state_size)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Flatten()(x)

    # แยกออกเป็น State Value และ Advantage
    state_value = tf.keras.layers.Dense(128, activation='relu')(x)
    state_value = tf.keras.layers.Dense(1, activation='linear')(state_value)

    advantage = tf.keras.layers.Dense(128, activation='relu')(x)
    advantage = tf.keras.layers.Dense(action_size, activation='linear')(advantage)

    # รวม State Value และ Advantage เพื่อให้ AI เลือก Action ที่เหมาะสมที่สุด
    q_values = tf.keras.layers.Lambda(lambda a: a[0] + (a[1] - tf.keras.backend.mean(a[1], axis=1, keepdims=True)))([state_value, advantage])
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# สร้างโมเดลหลักและ Target Network
model = build_dueling_dqn()
target_model = build_dueling_dqn()
target_model.set_weights(model.get_weights())

# สร้าง Stack สำหรับเก็บเฟรมย้อนหลัง
frame_stack = deque(maxlen=num_frames)

def reset_stacked_state():
    """รีเซ็ต Frame Stack เมื่อเริ่มเกมใหม่"""
    global frame_stack
    frame_stack.clear()

def get_stacked_state():
    """คืนค่า state ที่รวมเฟรมย้อนหลังและทิศทางลูก"""
    state, ballXSpeed, ballYSpeed = game.get_state()
    
    # แปลงเป็น numpy array และเพิ่ม channel สำหรับทิศทาง
    state = np.array(state)
    direction_channel = np.zeros_like(state)
    direction_channel[0, 0] = ballXSpeed  # -1 = ซ้าย, 1 = ขวา
    direction_channel[0, 1] = ballYSpeed  # -1 = ขึ้น, 1 = ลง
    
    state = np.stack([state, direction_channel], axis=-1)
    state = state.astype(np.float32) / 2  # Normalize

    if len(frame_stack) < num_frames:
        for _ in range(num_frames):
            frame_stack.append(state)

    frame_stack.append(state)
    stacked_state = np.concatenate(frame_stack, axis=-1)
    return stacked_state

def get_action(state, episode):
    """เลือก Action โดยใช้ Epsilon-Greedy Policy"""
    if np.random.rand() <= epsilon_decay(episode):
        return random.choice([0, 1, 2])  # สุ่ม Action
    return np.argmax(model.predict(state.reshape(1, *state_size), verbose=0))

def train_dqn(episodes=5000):
    global epsilon
    best_model_score = -float('inf')

    for episode in range(episodes):
        reset_stacked_state()
        state = get_stacked_state()
        done = False
        total_reward = 0
        hits = 0  # นับจำนวนครั้งที่ตีโดน

        while not done:
            action = get_action(state, episode)
            if action == 0:
                game.move_left()
            elif action == 2:
                game.move_right()

            hit, game_over = game.tick()
            next_state = get_stacked_state()

            # ปรับปรุง Reward Function
            reward = 10 if hit else -100 if game_over else 1
            total_reward += reward

            memory.append((state, action, reward, next_state, game_over))
            state = next_state

            game.render()
            print(f"Episode {episode}: Score = {total_reward}, Hits = {hits}, Epsilon = {epsilon_decay(episode):.4f}")
            
            if game_over:
                game.reset()
                reset_stacked_state()
                break

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # บันทึกโมเดลที่ดีที่สุด
        if avg_score > best_model_score:
            best_model_score = avg_score
            model.save("ai_pong_best.h5")
            print(f"💾 บันทึกโมเดลที่ดีที่สุด (Score: {avg_score:.2f})")

        print(f"Episode {episode}: Score = {total_reward}, Hits = {hits}, Epsilon = {epsilon_decay(episode):.4f}")

if __name__ == "__main__":
    train_dqn(episodes=5000)
    model.save("ai_pong_final.h5")
