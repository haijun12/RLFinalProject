import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
from craftium import CraftiumEnv

# env = gym.make("Craftium/ChopTree-v0")

env = CraftiumEnv(
    env_dir="/Users/asze01/Documents/craftium/craftium-envs/chop-tree",
    render_mode="human",
    obs_width=64,
    obs_height=64,
    rgb_observations=True,
    gray_scale_keepdim=False,
)

observation, info = env.reset()

for t in range(10000):
    action = env.action_space.sample()

    # plot the observation
    plt.clf()
    plt.imshow(observation)
    plt.pause(0.02)  # wait for 0.02 seconds

    observation, reward, terminated, truncated, _info = env.step(action)

    if terminated or truncated:
        print("done")
        observation, info = env.reset()

env.close()