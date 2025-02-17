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
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
learning_rate = 0.001
batch_size = 32
max_memory_size = 10000
tau = 0.01
memory = deque(maxlen=max_memory_size)
scores = []
num_frames = 4

state_size = (game.sceneSize, game.sceneSize, num_frames * 2)  # *2 เพราะมี speed channel
action_size = 3

# โหลดโมเดลที่บันทึกไว้
latest_model = None
latest_episode = 0
for i in range(100000, 0, -1000):
    filename = f"ai_pong_{i}.h5"
    if os.path.exists(filename):
        latest_model = filename
        latest_episode = i
        break

if latest_model:
    print(f"🔄 โหลดโมเดลจาก {latest_model} (เทรนไปแล้ว {latest_episode} episodes)")
    model = tf.keras.models.load_model(latest_model)
    epsilon = max(epsilon_min, epsilon * (epsilon_decay ** latest_episode))
else:
    print("🚀 เริ่มเทรนใหม่ ตั้งแต่ Episode 0")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=state_size),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  jit_compile=True)

# สร้าง Target Network
target_model = tf.keras.models.clone_model(model)
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
    # เก็บทิศทางที่มุมบนซ้าย - ใช้ค่าจริงเพื่อให้ AI รู้ทิศทางแม่นยำ
    direction_channel[0, 0] = ballXSpeed  # -1 = ซ้าย, 1 = ขวา
    direction_channel[0, 1] = ballYSpeed  # -1 = ขึ้น, 1 = ลง
    
    state = np.stack([state, direction_channel], axis=-1)
    state = state.astype(np.float32) / 2  # Normalize

    if len(frame_stack) < num_frames:
        for _ in range(num_frames):
            frame_stack.append(state)

    frame_stack.append(state)
    stacked_state = np.concatenate(frame_stack, axis=-1)
    return stacked_state, ballXSpeed, ballYSpeed

def get_action(state):
    if np.random.rand() <= epsilon:
        return random.choice([0, 1, 2])
    return np.argmax(model.predict(state.reshape(1, game.sceneSize, game.sceneSize, num_frames * 2), verbose=0))

def replay():
    """ใช้ Mini-Batch Training"""
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    
    states = np.array([sample[0] for sample in batch])
    actions = np.array([sample[1] for sample in batch])
    rewards = np.array([sample[2] for sample in batch])
    next_states = np.array([sample[3] for sample in batch])
    dones = np.array([sample[4] for sample in batch])

    targets = model.predict(states, verbose=0)
    next_q_values = target_model.predict(next_states, verbose=0)

    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += gamma * np.max(next_q_values[i])
        targets[i][actions[i]] = target

    model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)

def plot_training_results():
    """บันทึกกราฟแสดงผลการเทรน"""
    plt.figure(figsize=(12, 4))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.close()

def train_dqn(episodes=5000):
    global epsilon
    max_score = -float('inf')
    best_model_score = -float('inf')

    for episode in range(latest_episode + 1, latest_episode + episodes + 1):
        reset_stacked_state()
        state, ballXSpeed, ballYSpeed = get_stacked_state()
        done = False
        total_reward = 0
        hits = 0  # นับจำนวนครั้งที่ตีโดน

        while not done:
            action = get_action(state)

            if action == 0:
                game.move_left()
            elif action == 2:
                game.move_right()

            hit, game_over = game.tick()
            next_state, next_ballXSpeed, next_ballYSpeed = get_stacked_state()

            # คำนวณ reward
            reward = 0
            if hit:
                hits += 1
                reward = 10  # รางวัลพื้นฐานสำหรับการตีโดน
                if ballYSpeed > 0:  # ถ้าลูกกำลังลงมา แล้วตีได้ = ดี
                    reward += 5
            elif game_over:
                reward = -100 + hits
                done = True
            else:
                if ballYSpeed > 0:  # ลูกกำลังลงมา
                    # ให้ reward ตามตำแหน่งของ paddle เทียบกับลูก
                    paddle_center = game.paddleX + (game.paddleWidth / 2)
                    ball_distance = abs(paddle_center - game.ballX)
                    reward = 1 - (ball_distance / game.sceneSize)
                elif ballYSpeed < 0:  # ลูกกำลังขึ้นไปหากำแพง
                    reward = 3  # ให้รางวัลเพราะแสดงว่าเราตีลูกได้ดี

            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            if len(memory) >= batch_size:
                replay()

            state = next_state
            ballXSpeed = next_ballXSpeed
            ballYSpeed = next_ballYSpeed

            game.render()
            print(f"Episode {episode}: Score = {total_reward}, Hits = {hits}, Epsilon = {epsilon:.4f}")
            
            if game_over:
                game.reset()
                reset_stacked_state()

        scores.append(total_reward)

        # คำนวณคะแนนเฉลี่ย 100 episodes ล่าสุด
        current_mean = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # บันทึกโมเดลที่ดีที่สุด
        if current_mean > best_model_score:
            best_model_score = current_mean
            model.save("ai_pong_best.h5")
            print(f"💾 บันทึกโมเดลที่ดีที่สุด (Score: {current_mean:.2f})")

        if episode % 1000 == 0:
            model.save(f"ai_pong_{episode}.h5")
            np.savetxt("scores_log.txt", scores, fmt="%d")
            print(f"💾 บันทึกโมเดลที่ Episode {episode}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode}: Score = {total_reward}, Hits = {hits}, Epsilon = {epsilon:.4f}")

if __name__ == "__main__":
    train_dqn(episodes=5000)
    model.save("ai_pong_final.h5")
    plot_training_results()