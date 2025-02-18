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
    def __init__(self, state_size=15, action_size=3, memory_size=2000):
        # State size = 5 features * 3 frames
        self.state_size = state_size  
        self.action_size = action_size  # [moveLeft, stay, moveRight]
        
        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Memory for experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Neural Network for Deep Q-learning
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
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

        # Predict Q-values for current states and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target[i][action] = reward
            else:
                target[i][action] = reward + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def process_state(game_state, history_buffer):
    """
    Process current game state and maintain history buffer
    Returns flattened state vector combining current and historical states
    """
    current_state = np.array([
        game_state['ball_x'],
        game_state['ball_y'],
        game_state['ball_speed_x'],
        game_state['ball_speed_y'],
        game_state['paddle_x']
    ])
    
    history_buffer.append(current_state)
    if len(history_buffer) > 3:  # Keep last 3 frames
        history_buffer.popleft()
        
    # Flatten the history buffer into a single vector
    return np.concatenate(list(history_buffer))

def train_agent(game, episodes=1000):
    agent = PongDQNAgent()
    history_buffer = deque(maxlen=3)


    checkpoint_path = "latest_model.weights.h5"
    if os.path.exists(checkpoint_path):
        agent.model.load_weights(checkpoint_path)
        agent.update_target_model()
        print("Resumed model from checkpoint:", checkpoint_path)
    
    # Initialize history buffer with initial state
    initial_state = process_state(game.get_state(), history_buffer)
    for _ in range(3):  # Fill buffer with initial state
        history_buffer.append(np.zeros(5))
    
    for episode in range(episodes):
        # Reset game and history buffer
        game.reset()
        history_buffer.clear()
        for _ in range(3):
            history_buffer.append(np.zeros(5))
        
        total_reward = 0
        done = False
        
        while not done:
            # Get current state
            current_state = process_state(game.get_state(), history_buffer)
            
            # Choose action
            action = agent.act(current_state)
            
            # Execute action
            if action == 0:
                game.moveLeft()
            elif action == 2:
                game.moveRight()
            # action 1 is "stay" - do nothing
            
            # Progress game and get reward
            hit, done = game.tick()
            game.render()
            print(f"Episode: {episode + 1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

            reward = 1 if hit else (-10 if done else 0)
            total_reward += reward
            
            # Get new state
            next_state = process_state(game.get_state(), history_buffer)
            
            # Store experience
            agent.remember(current_state, action, reward, next_state, done)
            
            # Train agent
            agent.replay(32)
            
            if done:
                # Update target network every episode
                agent.update_target_model()
                print(f"Episode: {episode + 1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if (episode + 1) % 100 == 0:
            agent.model.save_weights(checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")

    #Final model saved as model_final.h5
    agent.model.save("model_final.h5")

    return agent

# Example usage:
"""
# Assuming we have a game class that implements the required interface:
game = PongGame()  # Your game implementation
agent = train_agent(game)

# To use the trained agent:
def play_game(agent, game):
    done = False
    history_buffer = deque(maxlen=3)
    for _ in range(3):
        history_buffer.append(np.zeros(5))
    
    while not done:
        state = process_state(game.get_state(), history_buffer)
        action = agent.act(state)
        
        if action == 0:
            game.moveLeft()
        elif action == 2:
            game.moveRight()
            
        hit, done = game.tick()
"""

game = PongGame(24,24,4)  
agent = train_agent(game)