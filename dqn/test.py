import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from craftium import CraftiumEnv
# /opt/homebrew/bin/python3.11 test.py
env = CraftiumEnv(
    env_dir="/Users/haijun/Desktop/Arcs/Spring 2025/ReinforcementLearning/Minecraft/craftium/craftium-envs/chop-tree",
    render_mode="human",
    obs_width=512,
    obs_height=512,
    rgb_observations=True,
    gray_scale_keepdim=False,
)

observation, info = env.reset()
print(info) 

plt.figure(figsize=(10, 7))
print(type(env.action_space))
for t in range(2000):
    action = env.action_space.sample()
    # print(action)

    # plot the observation
    plt.clf()
    plt.imshow(observation)
    plt.title(f"Timestep {t}")
    plt.pause(0.02)  # wait for 0.02 seconds

    observation, reward, terminated, truncated, _info = env.step(action)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close(clear=True)

