import game
import tensorflow as tf
import numpy as np
import random
import time
import os
from collections import deque

# ✅ ปิด Log ของ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ✅ ตรวจสอบว่าใช้ GPU ได้ไหม
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ ใช้ GPU สำหรับ Training!")
else:
    print("⚠️ ไม่พบ GPU, ใช้ CPU แทน")

# **พารามิเตอร์ของ RL**
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.001
batch_size = 32
max_memory_size = 10000
tau = 0.01  # Soft Update Rate
memory = deque(maxlen=max_memory_size)
scores = []
num_frames = 4  # ✅ กำหนดจำนวนเฟรมย้อนหลัง

state_size = (game.sceneSize, game.sceneSize, num_frames)
action_size = 3

# **โหลดโมเดลที่บันทึกไว้**
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
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  jit_compile=True)

# **สร้าง Target Network**
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

# **สร้าง Stack สำหรับเก็บเฟรมย้อนหลัง**
frame_stack = deque(maxlen=num_frames)

def reset_stacked_state():
    """ ✅ รีเซ็ต Frame Stack เมื่อเริ่มเกมใหม่ """
    global frame_stack
    frame_stack.clear()

def get_stacked_state():
    """ ✅ คืนค่า state ที่รวมเฟรมย้อนหลัง num_frames เฟรม """
    state, ballXSpeed, ballYSpeed = game.get_state()
    state = np.expand_dims(state, axis=-1).astype(np.float32) / 2  # Normalize

    if len(frame_stack) < num_frames:
        for _ in range(num_frames):
            frame_stack.append(state)

    frame_stack.append(state)
    stacked_state = np.concatenate(frame_stack, axis=-1)
    return stacked_state, ballXSpeed, ballYSpeed

def get_action(state):
    if np.random.rand() <= epsilon:
        return random.choice([0, 1, 2])
    return np.argmax(model.predict(state.reshape(1, game.sceneSize, game.sceneSize, num_frames), verbose=0))

def replay():
    """ ✅ ใช้ Mini-Batch Training """
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

def train_dqn(episodes=5000):
    global epsilon
    for episode in range(latest_episode + 1, latest_episode + episodes + 1):
        reset_stacked_state()  
        state, ballXSpeed, ballYSpeed = get_stacked_state()
        done = False
        total_reward = 0

        while not done:
            action = get_action(state)

            if action == 0:
                game.move_left()
            elif action == 2:
                game.move_right()

            hit, game_over = game.tick()
            next_state, next_ballXSpeed, next_ballYSpeed = get_stacked_state()

            # ✅ ปรับค่า Reward ให้ AI
            reward = 1  # ปกติ
            if hit:
                reward = 10  # ตีโดนได้แต้ม
            if game_over:
                reward = -100  # แพ้เกม
                done = True

            total_reward += reward  # ✅ AI ได้รับรางวัล

            memory.append((state, action, reward, next_state, done))
            if len(memory) >= batch_size:
                replay()  # ✅ Train AI

            state = next_state

            game.render()
            print(f"Episode {episode}: Score = {total_reward}, Epsilon = {epsilon:.4f}")
            
            if game_over:
                game.reset()
                reset_stacked_state()

        scores.append(total_reward)  # ✅ บันทึกคะแนน

        # ✅ หยุดการเทรนถ้า AI เล่นได้ดี
        if len(scores) > 100 and np.mean(scores[-100:]) > 200:
            print("🎉 AI ฉลาดแล้ว! หยุดการฝึก")
            break

        if episode % 1000 == 0:
            model.save(f"ai_pong_{episode}.h5")
            np.savetxt("scores_log.txt", scores, fmt="%d")
            print(f"💾 บันทึกโมเดลที่ Episode {episode}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode}: Score = {total_reward}, Epsilon = {epsilon:.4f}")

if __name__ == "__main__":
    train_dqn(episodes=5000)
    model.save("ai_pong_final.h5")
