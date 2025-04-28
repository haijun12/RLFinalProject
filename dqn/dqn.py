import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from craftium import CraftiumEnv
import random
from collections import deque

class ActionMapping:
    def __init__(self):
        # Define the discrete actions we want to support
        self.actions = [
            # Movement actions
            {'forward': 1},                               # 0: Move forward
            {'backward': 1},                              # 1: Move backward
            {'left': 1},                                  # 2: Turn left
            {'right': 1},                                 # 3: Turn right
            # {'jump': 1},                                  # 4: Jump
            
            # Look actions (using mouse)
            {'mouse': np.array([0.3, 0])},                # 5: Look right
            {'mouse': np.array([-0.3, 0])},               # 6: Look left
            {'mouse': np.array([0, 0.3])},                # 7: Look up
            {'mouse': np.array([0, -0.3])},               # 8: Look down
            
            # Tool actions
            {'dig': 1},                                   # 9: Dig/mine
            # {'place': 1},                                 # 10: Place block
            
            # Slot selection
            # {'slot_1': 1},                                # 11: Select slot 1
            # {'slot_2': 1},                                # 12: Select slot 2
            # {'slot_3': 1},                                # 13: Select slot 3
            # {'slot_4': 1},                                # 14: Select slot 4
            # {'slot_5': 1},                                # 15: Select slot 5
            # {'slot_6': 1},                                # 16: Select slot 6
            # {'slot_7': 1},                                # 17: Select slot 7
            # {'slot_8': 1},                                # 18: Select slot 8
            # {'slot_9': 1},                                # 19: Select slot 9
            # Combined actions
            {'forward': 1, 'dig': 1},                     # 14: Move forward while digging
            {'forward': 1, 'jump': 1},                    # 15: Jump forward
            {'forward': 1, 'mouse': np.array([0.3, 0])},  # 16: Move forward while looking right
            {'forward': 1, 'mouse': np.array([-0.3, 0])}, # 17: Move forward while looking left
        ]
        
        # Number of available actions
        self.n_actions = len(self.actions)
    
    def get_action_dict(self, action_idx):
        """Convert action index to action dictionary"""
        base_dict = {
            'aux1': 0, 'backward': 0, 'dig': 0, 'drop': 0, 'forward': 0,
            'inventory': 0, 'jump': 0, 'left': 0, 'place': 0, 'right': 0,
            'slot_1': 0, 'slot_2': 0, 'slot_3': 0, 'slot_4': 0, 'slot_5': 0,
            'slot_6': 0, 'slot_7': 0, 'slot_8': 0, 'slot_9': 0, 'sneak': 0, 'zoom': 0,
            'mouse': np.array([0.0, 0.0])
        }
        
        # Update with the selected action
        selected_action = self.actions[action_idx]
        for k, v in selected_action.items():
            base_dict[k] = v
            
        return base_dict
    
    def describe_action(self, action_idx):
        """Return a human-readable description of the action"""
        action_descriptions = [
            "Move forward", "Move backward", "Turn left", "Turn right", 
            # "Jump",
            "Look right", "Look left", "Look up", "Look down",
            "Dig/mine", 
            # "Place block",
            # "Select slot 1", "Select slot 2", "Select slot 3", "Select slot 4", "Select slot 5",
            # "Select slot 6", "Select slot 7", "Select slot 8", "Select slot 9",
            "Move forward while digging", "Jump forward", 
            "Move forward while looking right", "Move forward while looking left",
        ]
        return action_descriptions[action_idx]
    

class NN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Conv layer 1: input channels = 3 (RGB), output channels = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        
        # Conv layer 2: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)

        # Conv layer 3: 32 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        # After conv3 we have (batch, 64, 5, 5) so flattened it's 64*5*5 = 1600
        self.fc1 = nn.Linear(1600, 128)  # first dense layer
        self.fc2 = nn.Linear(128, n_actions)  # final layer outputs Q-values
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 16, 30, 30)
        x = F.relu(self.conv2(x))  # (batch, 32, 13, 13)
        x = F.relu(self.conv3(x))  # (batch, 64, 5, 5)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # output raw Q-values
    
    def visualize_layers(self, x,  layers=("conv1", "conv2", "conv3")):
        """
        Visualize outputs from specified layers.
        `x` should be a (1, 3, 64, 64) input tensor.
        """
        activations = {}
        x = x.clone().detach()  # Make sure no gradient is tracked
        with torch.no_grad():
            if "conv1" in layers:
                x = F.relu(self.conv1(x))
                activations["conv1"] = x.clone()
            if "conv2" in layers:
                x = F.relu(self.conv2(x))
                activations["conv2"] = x.clone()
            if "conv3" in layers:
                x = F.relu(self.conv3(x))
                activations["conv3"] = x.clone()
        
        fig, axes = plt.subplots(nrows=3, ncols=16, figsize=(16, 8))
        axes = axes.flatten()
        # Plot each requested layer
        plt_index = 0
        for name, act in activations.items():
            # print(name)
            num_channels = act.shape[1]
            for i in range(min(16, num_channels)):  # Show up to 8 feature maps
                axes[plt_index].imshow(act[0, i].cpu(), cmap="viridis")
                axes[plt_index].axis("off")
                axes[plt_index].set_title(f"{i}")
                plt_index += 1
        plt.tight_layout()
        plt.show()

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, state_dim, action_mapper, device="cuda" if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_mapper = action_mapper
        self.device = device
        print(f"Using device: {device}")
        # Define off policy target and policy networks, both set to a specific device
        self.policy_net = NN(action_mapper.n_actions).to(device)
        self.target_net = NN(action_mapper.n_actions).to(device)
        # Set target net to same weights as policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target net only evlauates
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        # Experience replay
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 10  # How often to update target network
        self.batch_size = 32
        
        self.steps = 0

    def preprocess(self, observation):
        """Convert observation to tensor and normalize"""
        if isinstance(observation, np.ndarray):
            # Convert from (H, W, C) to (C, H, W) and add batch dimension
            observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        return observation.to(self.device)
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_mapper.n_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
        
        # Convert to action dictionary
        action_dict = self.action_mapper.get_action_dict(action_idx)
        
        return action_idx, action_dict
    
    def update_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network by copying the policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self):
        """Perform one training step with a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples
        
        # print("Training step...", len(self.memory))
        
        # Sample from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.cat([self.preprocess(s) for s in states])
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.cat([self.preprocess(s) for s in next_states])
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Q values for current states
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Calculate loss
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target_network()
            
        return loss.item()
    
# Step 3: Training loop
def train_dqn(env, agent, num_episodes=1000, max_steps=500):
    rewards_history = []
    action_counts = [0] * agent.action_mapper.n_actions
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.preprocess(obs)
        episode_reward = 0
        losses = []
        episode_actions = []
        
        for step in range(max_steps):
            # Select action
            action_idx, action_dict = agent.select_action(state)
            action_counts[action_idx] += 1
            episode_actions.append(action_idx)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            next_state = agent.preprocess(next_obs)
            if reward > 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f} Successfully mined a tree!" )
                reward = 100 # Give a larger reward for successful mining
            
            # Store transition in replay memory
            agent.memory.push(obs, action_idx, reward, next_obs, done)
            
            # Optimize model
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # If episode ended
            if done:
                print("Finished an episode (Successfully mined a tree)")
                break

        # print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Update exploration rate
        
        # Log results
        rewards_history.append(episode_reward)
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if episode % 10 == 0:
            agent.update_epsilon()
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            # Print most common actions in this episode
            if episode_actions:
                unique, counts = np.unique(episode_actions, return_counts=True)
                top_actions = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:3]
                print("Top actions this episode:")
                for action_idx, count in top_actions:
                    print(f"  {agent.action_mapper.describe_action(action_idx)}: {count} times")
        
        # Plot progress every 100 episodes
        # if episode % 100 == 0 and episode > 0:
        #     plot_progress(rewards_history)
            
        #     # Plot action distribution
        #     plt.figure(figsize=(12, 6))
        #     plt.bar(range(len(action_counts)), action_counts)
        #     plt.xlabel('Action Index')
        #     plt.ylabel('Count')
        #     plt.title('Action Distribution')
        #     plt.xticks(range(len(action_counts)), 
        #               [f"{i}: {agent.action_mapper.describe_action(i)}" for i in range(len(action_counts))],
        #               rotation=90)
        #     plt.tight_layout()
        #     plt.savefig(f'action_dist_ep_{episode}.png')
        #     plt.show()
    
    return rewards_history, action_counts

def plot_progress(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f'dqn_rewards_ep_{len(rewards)}.png')
    plt.show()
    
if __name__ == "__main__":
    # Load the environment
    env = CraftiumEnv(
        env_dir="/Users/haijun/Desktop/Arcs/Spring 2025/ReinforcementLearning/Minecraft/craftium/craftium-envs/chop-tree",
        obs_width=64,
        obs_height=64,
        render_mode=None,  # Disable rendering
        rgb_observations=True,
        gray_scale_keepdim=False,
    )
    
    # Get observation space dimensions
    observation, info = env.reset()
    
    # Optional: Visualize initial observation
    # plt.figure(figsize=(6, 6))
    # plt.imshow(observation)
    # plt.title("Initial Observation")
    # plt.show()
    
    # Create action mapping
    action_mapper = ActionMapping()
    print(f"Total number of actions: {action_mapper.n_actions}")
    for i in range(action_mapper.n_actions):
        print(f"{i}: {action_mapper.describe_action(i)}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=(3, 64, 64),  # RGB image (channels, height, width)
        action_mapper=action_mapper
    )
    
    # Train the agent
    rewards, action_counts = train_dqn(env, agent, num_episodes=500, max_steps=500)
    
    # Save trained model
    torch.save(agent.policy_net.state_dict(), "dqn_minecraft_model.pth")
    
    # Plot final results
    plot_progress(rewards)
    
    # Plot action distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(action_counts)), action_counts)
    plt.xlabel('Action Index')
    plt.ylabel('Count')
    plt.title('Action Distribution')
    plt.xticks(range(len(action_counts)), 
              [f"{i}: {action_mapper.describe_action(i)}" for i in range(len(action_counts))],
              rotation=90)
    plt.tight_layout()  
    plt.savefig('final_action_dist.png')
    plt.show()
    
    # Test the trained agent
    print("Testing trained agent...")
    obs, _ = env.reset()
    state = agent.preprocess(obs)
    done = False
    total_reward = 0
    test_actions = []
    
    while not done:
        action_idx, action_dict = agent.select_action(state, training=False)
        test_actions.append(action_idx)
        obs, reward, terminated, truncated, _ = env.step(action_dict)
        done = terminated or truncated
        state = agent.preprocess(obs)
        total_reward += reward
    
    # print(f"Test episode finished with reward: {total_reward}")
    
    # Print action distribution in test episode
    unique, counts = np.unique(test_actions, return_counts=True)
    print("\nAction distribution during testing:")
    for action_idx, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        print(f"  {action_mapper.describe_action(action_idx)}: {count} times")
    
    env.close()

# # Load the environment
# env = CraftiumEnv(
#     env_dir="/Users/haijun/Desktop/Arcs/Spring 2025/ReinforcementLearning/Minecraft/craftium/craftium-envs/chop-tree",
#     render_mode="human",
#     obs_width=64,
#     obs_height=64,
#     rgb_observations=True,
#     gray_scale_keepdim=False,
# )

# observation, info = env.reset()
# plt.figure(figsize=(15, 8))
# plt.suptitle("Original Image", fontsize=16)
# plt.imshow(observation)
# model = NN(n_actions=10)
# # Normalize and prepare observation
# obs_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

# # Put model in eval mode for visualization
# model.eval()

# # Visualize
# model.visualize_layers(obs_tensor, layers=["conv1", "conv2", "conv3"])

# model.train()