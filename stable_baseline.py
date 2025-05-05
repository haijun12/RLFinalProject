from stable_baselines3 import DQN
from craftium import CraftiumEnv
from craftium.wrappers import DiscreteActionWrapper
import matplotlib.pyplot as plt
import torch
import gym
import numpy as np
import cv2

class EdgeEffectWrapper(gym.ObservationWrapper):
    def __init__(self, env, edge_width=8, effect='blur', blur_strength=5):
        """
        env: gym environment
        edge_width: width of the edge region to apply the effect
        effect: 'blur' or 'black'
        blur_strength: kernel size for Gaussian blur (must be odd)
        """
        super().__init__(env)
        assert effect in ['blur', 'black'], "effect must be 'blur' or 'black'"
        self.edge_width = edge_width
        self.effect = effect
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1  # Ensure odd

    def observation(self, obs):
        obs_img = obs.squeeze()  # (H, W)
        mask = np.ones_like(obs_img, dtype=np.float32)

        h, w = obs_img.shape
        ew = self.edge_width

        # Create edge mask
        mask[:ew, :] = 0
        mask[-ew:, :] = 0
        mask[:, :ew] = 0
        mask[:, -ew:] = 0

        if self.effect == 'blur':
            blurred_img = cv2.GaussianBlur(obs_img, (self.blur_strength, self.blur_strength), 0)
            processed = obs_img * mask + blurred_img * (1 - mask)
        elif self.effect == 'black':
            processed = obs_img * mask  # Simply zero out the edges

        return processed[:, :, np.newaxis]  # (H, W, 1)


env = CraftiumEnv(
    env_dir="/Users/haijun/Desktop/Arcs/Spring 2025/ReinforcementLearning/Minecraft/craftium/craftium-envs/chop-tree",
    render_mode="human",
    seed=2000,
    obs_width=64,
    obs_height=64,
    rgb_observations=False,
    gray_scale_keepdim=True,
    frameskip=4,
    max_timesteps=1000,
)
env= DiscreteActionWrapper(
    env,
    actions=[
        "forward", "jump", "dig", "mouse x+",
        "mouse x-", "mouse y+", "mouse y-"
    ],
    mouse_mov=0.2,
)
env = EdgeEffectWrapper(env, edge_width=8, effect='blur', blur_strength=5)
env = EdgeEffectWrapper(env, edge_width=8, effect='black')

# model = DQN("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=100000, log_interval=4,
#             progress_bar=True)
# model.save("choptree_noterm")
# del model

model = DQN.load("choptree_noterm")
obs, info = env.reset()
print(obs.shape)
plt.figure(figsize=(10, 7))
for t in range(1000):
    action, _states= model.predict(obs, deterministic=True)
    action = torch.tensor([[int(action)]], device='mps')
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Episode finished after {} timesteps".format(t + 1))
        break
    plt.clf()
    plt.imshow(obs)
    plt.title(f"Timestep {t}")
    plt.pause(0.02)  # wait for 0.02 seconds

