# src/agents/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.to(device)
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, config, device):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = device
        
        # Q-Networks
        self.q_network = DQNNetwork(state_size, action_size, device)
        self.target_network = DQNNetwork(state_size, action_size, device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch and move to device
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * max_next_q_values
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

