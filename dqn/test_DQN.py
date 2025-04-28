import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import craftium
from craftium import CraftiumEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_observation(obs):
    obs = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1) 
    return obs.unsqueeze(0)

def map_discrete_to_dict(action_index):
    valid_keys = ['forward', 'jump', 'dig']
    action_dict = {k: 0 for k in valid_keys}
    mouse = [0.0, 0.0]

    if action_index == 1:
        action_dict['forward'] = 1
    elif action_index == 2:
        action_dict['jump'] = 1
    elif action_index == 3:
        action_dict['dig'] = 1
    elif action_index == 4:
        mouse[0] = 1.0  
    elif action_index == 5:
        mouse[0] = -1.0  
    elif action_index == 6:
        mouse[1] = 1.0  
    elif action_index == 7:
        mouse[1] = -1.0  

    action_dict_full = {k: 0 for k in [
        "forward", "backward", "left", "right", "jump", "aux1", "sneak",
        "zoom", "dig", "place", "drop", "inventory", "slot_1", "slot_2",
        "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8", "slot_9"]}
    action_dict_full.update(action_dict)
    action_dict_full['mouse'] = np.array(mouse, dtype=np.float32)
    return action_dict_full

env = CraftiumEnv(
    env_dir="/Users/asze01/Documents/craftium/craftium-envs/chop-tree",
    render_mode=None, #change back to human later
    obs_width=64,
    obs_height=64,
    rgb_observations=True # change to grayscale so it runs faster
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = 8  
h, w, _ = env.observation_space.shape

policy_net = DQN(h, w, n_actions).to(device)
target_net = DQN(h, w, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

episode_durations = []
num_episodes = 2 #change back to - 600 if torch.cuda.is_available() else 50
total_reward = 0.0

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess_observation(state)
    for t in range(100): # remove later
        for t in count():
            print(f"Episode {i_episode + 1}/{num_episodes} — Steps: {t + 1} — Total transitions: {len(memory)}")
            action = select_action(state)
            env_action = map_discrete_to_dict(action.item())
            obs, reward, terminated, truncated, _ = env.step(env_action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = preprocess_observation(obs) if not done else None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            total_reward += reward.item()

            if done:
                episode_durations.append(t + 1)
                print(f"Episode {i_episode + 1}/{num_episodes} — Steps: {t + 1} — Total transitions: {len(memory)}") 
                print(f"Episode {i_episode+1}/{num_episodes} | Steps: {t+1} | Total Reward: {total_reward:.2f}")
                break

print('Training complete')
env.close()
