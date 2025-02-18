import os
import numpy as np
import tensorflow as tf
from collections import deque
import random
from game import PongGame

# ตรวจสอบว่าใช้ GPU ได้ไหม
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ ใช้ GPU สำหรับ Training!")
else:
    print("⚠️ ไม่พบ GPU, ใช้ CPU แทน")

class PongDQNAgent:
    def __init__(self, state_size=15, action_size=3, memory_size=5000):
        self.state_size = state_size  
        self.action_size = action_size  # [moveLeft, stay, moveRight]
        
        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        
        # Memory สำหรับ experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Neural Network สำหรับ Deep Q-learning (เพิ่มความซับซ้อน)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state

        # คาดการณ์ Q-values สำหรับสถานะปัจจุบันและสถานะถัดไป
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target[i][action] = reward
            else:
                target[i][action] = reward + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)
        # ลบการลด epsilon ออกจาก replay เพื่อให้ลดทีละ episode แทน

def process_state(game_state, history_buffer):
    """
    Process สถานะเกมปัจจุบันและอัปเดต history buffer
    คืนค่าเป็นเวกเตอร์ที่รวมข้อมูลของ frame ปัจจุบันและ frame ก่อนหน้า
    """
    current_state = np.array([
        game_state['ball_x'],
        game_state['ball_y'],
        game_state['ball_speed_x'],
        game_state['ball_speed_y'],
        game_state['paddle_x']
    ])
    
    history_buffer.append(current_state)
    if len(history_buffer) > 3:  # เก็บเฉพาะ 3 frame ล่าสุด
        history_buffer.popleft()
        
    # Flatten history buffer เป็นเวกเตอร์เดียว
    return np.concatenate(list(history_buffer))

def train_agent(game, episodes=1000):
    agent = PongDQNAgent()
    history_buffer = deque(maxlen=3)

    checkpoint_path = "latest_model.weights.h5"
    if os.path.exists(checkpoint_path):
        agent.model.load_weights(checkpoint_path)
        agent.update_target_model()
        print("Resumed model from checkpoint:", checkpoint_path)
    
    # เตรียม history buffer ด้วยค่าเริ่มต้น (3 frame ของศูนย์)
    for _ in range(3):
        history_buffer.append(np.zeros(5))
    
    for episode in range(episodes):
        # รีเซ็ตเกมและ history buffer ในแต่ละ episode
        game.reset()
        history_buffer.clear()
        for _ in range(3):
            history_buffer.append(np.zeros(5))
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # รับสถานะปัจจุบัน
            current_state = process_state(game.get_state(), history_buffer)
            
            # เลือก action โดยใช้ epsilon-greedy
            action = agent.act(current_state)
            
            # ดำเนินการ action ที่เลือก
            if action == 0:
                game.moveLeft()
            elif action == 2:
                game.moveRight()
            # action 1 คือ "stay" - ไม่ทำอะไร
            
            # ทำให้เกมก้าวไปข้างหน้าและรับผลลัพธ์
            hit, done = game.tick()
            game.render()
            step += 1
            print(f"Episode: {episode + 1}, Step: {step}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

            # Reward shaping: ให้รางวัลตามผลลัพธ์ของ action
            if done:
                reward = -10
            elif hit:
                reward = 1
            else:
                reward = -0.01  # ให้รางวัลติดลบเล็กน้อยเพื่อผลักดันให้ agent พยายามอยู่ในเกม
            total_reward += reward
            
            # รับสถานะถัดไป
            next_state = process_state(game.get_state(), history_buffer)
            
            # เก็บประสบการณ์
            agent.remember(current_state, action, reward, next_state, done)
            
            # ฝึก agent โดยใช้ experience replay
            agent.replay(32)
            
        # อัปเดท target network ทุกครั้งที่จบ episode
        agent.update_target_model()
        # ลดค่า epsilon ทีละ episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        print(f"Episode {episode + 1} finished with Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if (episode + 1) % 100 == 0:
            agent.model.save_weights(checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")

    # บันทึกโมเดลขั้นสุดท้ายเป็น model_final.h5
    agent.model.save("model_final.h5")

    return agent

# ตัวอย่างการใช้งาน:
# สมมุติว่าคุณมีคลาสเกม Pong ที่ implement อินเตอร์เฟสที่กำหนดไว้:
# game = PongGame()  # การ implement เกมของคุณ
# agent = train_agent(game)

game = PongGame(24, 24, 4)  
agent = train_agent(game, episodes=5000)
