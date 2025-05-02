import matplotlib.pyplot as plt
import numpy as np
from craftium import CraftiumEnv
import random
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from craftium.wrappers import DiscreteActionWrapper
from gymnasium.wrappers import FrameStack
import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # if frame stacked, the input format is (n-frames, h, w, c) instead of (h, w, c)
        if len(input_shape) == 3:
            # shape = (H, W, C)
            h, w, c = input_shape
            in_channels = c
        elif len(input_shape) == 4:
            # shape = (N_FRAMES, H, W, C)
            n_frames, h, w, c = input_shape
            in_channels = n_frames * c
        else:
            raise ValueError(f"Invalid input_shape: {input_shape}")

        # Convolutional layers — expects input as (channels, height, width)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Dynamically compute the output size of the conv layers
        with torch.no_grad():
            # torch format: (B, C, H, W)
            dummy_input = torch.zeros(1, in_channels, h, w)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            conv_output_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        # Output Q-values for each of the 8 actions
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values to [0, 1]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Output shape: (batch_size, n_actions)


class Agent():
    def __init__(self, input_shape, action_space, hyperparameters, device):
        self.input_shape = input_shape
        self.action_space = action_space
        self.device = device

        self.BATCH_SIZE = hyperparameters.get("BATCH_SIZE", 128)
        self.GAMMA = hyperparameters.get("GAMMA", 0.99)
        self.EPS_START = hyperparameters.get("EPS_START", 0.9)
        self.EPS_END = hyperparameters.get("EPS_END", 0.05)
        self.EPS_DECAY = hyperparameters.get("EPS_DECAY", 1000)
        self.TAU = hyperparameters.get("TAU", 0.005)
        self.LR = hyperparameters.get("LR", 1e-4)
        self.memory_size = hyperparameters.get("MEMORY_SIZE", 10000)

        self.memory = ReplayMemory(self.memory_size)
        self.policy_net = DQN(self.input_shape, self.action_space.n).to(device)
        self.target_net = DQN(self.input_shape, self.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.steps_done = 0
        self.eps_threshold = 0

    def reset(self):
        self.steps_done = 0
        self.eps_threshold = 0
        self.policy_net = DQN(self.input_shape, self.action_space.n).to(device)
        self.target_net = DQN(self.input_shape, self.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(self.memory_size)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.LR, amsgrad=True)

    def reshape_state(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        # add extra dimensions based on if frames are stacked or not
        if state.ndim == 3:
            state = state.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        elif state.ndim == 4:
            # Frame stacked grayscale, shape (N, H, W, C)
            state = state.permute(0, 3, 1, 2)  # (N, C, H, W)
            state = state.reshape(
                1, -1, state.shape[2], state.shape[3])  # (1, N*C, H, W)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        return state

    def decay_epsilon(self):
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, state, with_eps=True):
        sample = random.random()

        if not with_eps:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long)

    def train_agent(self):
        # only train if we have enough experience in memory buffer
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch. This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                          if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def load_policy_net(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))


class ChopTreeEnv():
    def __init__(self, config):
      # if framestack is true, we dont have additional dimension for color channel since the stacked frame is the 3rd dimension
        self.env = CraftiumEnv(
            env_dir=config.get("env_dir", "path/to/chop-tree"),
            render_mode=config.get("render_mode", "human"),
            seed=config.get("seed", 0),
            obs_width=config.get("obs_w", 64),
            obs_height=config.get("obs_h", 64),
            rgb_observations=config.get("obs_rgb", False),
            gray_scale_keepdim=True,
            frameskip=config.get("n_frame_skip", 4),
            max_timesteps=config.get("max_timesteps", 1000),
        )

        # Wrap with DiscreteActionWrapper
        self.env = DiscreteActionWrapper(
            self.env,
            actions=config.get("actions", [
                "forward", "jump", "dig", "mouse x+",
                "mouse x-", "mouse y+", "mouse y-"
            ]),
            mouse_mov=config.get("mouse_mov", 0.2),
        )

        # Wrap with FrameStack
        if config.get("frame_stack", False):
            self.env = FrameStack(
                self.env, num_stack=config.get("num_stack", 4))

    def __getattr__(self, name):
        # Delegate attribute lookup to the wrapped env
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def _penalize_timestep(self, reward):
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self._penalize_timestep(reward)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self, clear=False):
        return self.env.close()

    def get_env(self):
        return self.env


class TrainingLogger():
    def __init__(self):
        self.episode_durations = []
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = 0
        self.total_steps = 0
        self.episode_start_time = 0

    def start_episode(self):
        self.episode_start_time = time.time()
        self.episode_rewards = 0

    def print_current_episode(self):
        print(f"Current Episode: {self.current_episode}")
        print(f"Current timestep: {self.current_step}")
        print(f"Episode rewards: {self.episode_rewards}")
        print(
            f"Episode Elapsed Time: {time.time() - self.episode_start_time:.4f} seconds")
        print(f"Total steps trained: {self.total_steps}")
        print("-----------------------------")

    def reset(self):
        self.episode_durations = []
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = 0
        self.total_steps = 0


def train_agent(env: ChopTreeEnv, agent: Agent, num_episodes, logger: TrainingLogger, save_model=False, model_filename="model.pth"):
    agent.reset()
    env.reset()
    episode_rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment
        state, info = env.reset()
        state = agent.reshape_state(state)
        logger.current_episode = i_episode
        logger.start_episode()

        for t in count():
            logger.current_step = t
            action = agent.select_action(state)
            observation, reward, _, truncated, _ = env.step(
                action.item())
            done = truncated

            # log reward
            logger.episode_rewards += reward

            # if terminated:
            #     print("Episode finished after {} timesteps".format(t + 1))
            #     next_state = None
            # else:
            next_state = agent.reshape_state(observation)

            # Store the transition in memory
            reward = torch.tensor([reward], device=device)
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.train_agent()

            # Soft update target network
            agent.update_target_network()

            # decay epsilon
            agent.decay_epsilon()

            # agent record steps done
            agent.steps_done += 1
            logger.total_steps += 1

            if done:
                logger.episode_durations.append(t + 1)
                logger.print_current_episode()
                break
        # logger.episode_durations.append(t + 1)
        # logger.print_current_episode()
        episode_rewards.append(logger.episode_rewards)

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.show()
    if save_model:
        torch.save(agent.policy_net.state_dict(),
                   model_filename)


def test_agent(env: ChopTreeEnv, agent: Agent, max_time_steps, model_filepath, frame_stacked, logger: TrainingLogger):
    agent.load_policy_net(model_filepath)
    observation, info = env.reset()
    logger.start_episode()

    plt.figure(figsize=(10, 7))
    for t in range(max_time_steps):
        logger.current_step = t
        action = agent.select_action(
            agent.reshape_state(observation), with_eps=False)

        # plot the observation
        plt.clf()
        plt.imshow(observation[0] if frame_stacked else observation)
        plt.title(f"Timestep {t}")
        plt.pause(0.02)  # wait for 0.02 seconds

        observation, reward, terminated, truncated, _info = env.step(
            action)
        

        logger.episode_rewards += reward

        if terminated or truncated:
            plt.close()
            logger.print_current_episode()
            break
    env.close()


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# HYPERPARAMETERS
agent_hyperparameters = {
    "MEMORY_SIZE": 10000,
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 1000,
    "TAU": 0.005,
    "LR": 1e-4
}

# ENVIRONMENT SETTINGS
env_configs = {
    # "env_dir": "/Users/chengxi600/Documents/craftium/craftium-envs/chop-tree",
    "env_dir": "/Users/haijun/Desktop/Arcs/Spring 2025/ReinforcementLearning/Minecraft/craftium/craftium-envs/chop-tree",
    "obs_w": 64,
    "obs_h": 64,
    "obs_rgb": False,
    "n_frame_skip": 4,
    "render_mode": "human",
    "seed": 2025,
    "max_timesteps": 500,
    "frame_stack": True,
    "num_stack": 4,
}

# ENVIRONMENT
env = ChopTreeEnv(env_configs)

# AGENT
agent = Agent(env.observation_space.shape, env.action_space,
              agent_hyperparameters, device)

# TRAINING
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 100
else:
    num_episodes = 10

logger = TrainingLogger()

train_agent(env, agent, 100, logger, save_model=True,
            model_filename="dqn_model_stacked.pth")

# test_agent(env, agent, 1000, "dqn_model_stacked.pth",
#            env_configs["frame_stack"], logger)
