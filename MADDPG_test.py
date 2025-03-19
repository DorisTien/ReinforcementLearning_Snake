import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from snake_env import SnakeGameAI  # Import the environment

# Hyperparameters
GAMMA = 0.99
TAU = 0.01
LR_ACTOR = 0.001
LR_CRITIC = 0.001
BUFFER_SIZE = 100000
BATCH_SIZE = 64

# Neural Networks for Actor and Critic
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# MADDPG Agent
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critics]
        
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.loss_fn = nn.MSELoss()
        
        # Copy parameters to target networks
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
    
    def select_action(self, state1, state2):
        state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0)
        state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
        
        action1 = self.actors[0](state1).detach().numpy()[0]
        action2 = self.actors[1](state2).detach().numpy()[0]
        
        return np.argmax(action1), np.argmax(action2)  # Select the action with max probability

def train_maddpg(env, episodes=500):
    maddpg = MADDPGAgent(state_dim=11, action_dim=3, num_agents=2)
    scores1, scores2 = [], []
    episodes_list = list(range(1, episodes + 1))
    
    for episode in episodes_list:
        env.reset()
        state1 = env.get_state(1)
        state2 = env.get_state(2)
        
        done = False
        while not done:
            action1, action2 = maddpg.select_action(state1, state2)
            next_state1 = env.get_state(1)
            next_state2 = env.get_state(2)
            
            reward1 = random.randint(-10, 10)  # Placeholder reward
            reward2 = random.randint(-10, 10)  # Placeholder reward
            done = random.choice([True, False])  # Placeholder termination condition
            
            maddpg.memory.push(np.concatenate((state1, state2)), [action1, action2], [reward1, reward2], np.concatenate((next_state1, next_state2)), [done, done])
            state1, state2 = next_state1, next_state2
        
        scores1.append(reward1)
        scores2.append(reward2)
        print(f"Episode {episode}: Snake1 Score: {reward1}, Snake2 Score: {reward2}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, scores1, label="Snake 1 Score", linestyle='-', marker='.')
    plt.plot(episodes_list, scores2, label="Snake 2 Score", linestyle='-', marker='.')
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("MADDPG Training Progress for Snake AI")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    env = SnakeGameAI()
    train_maddpg(env)