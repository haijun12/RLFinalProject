Filename: single.py, init.lua
Languages: Python, Lua
Libraries: PyTorch, NumPy, Gymnasium, Matplotlib, Craftium, Minetest
By: Ash Sze, Haijun Si, Cheng Xi Tsou
For: CS 138 Reinforcement Learning

This project trains a Deep Q-Network (DQN) agent in the Craftium environment to learn the task of chopping down a tree. 
The environment is partially modded with Lua, with reward shaping logic via init.lua. The DQN training loop is  in Python through single.py.
Additionally, we evaluate how degraded visual input (e.g., blurred, cropped, edge-mapped) impacts learning performance.

single.py
single.py trains a DQN agent using raw pixel observations (64×64 grayscale) in a discrete action space.
It has for its agent architecture, a convolutional Q-network. The training loop uses experience replay, Polyak averaging, and epsilon-greedy exploration
We log results by live plotting of the reward and loss via matplotlib.

Hyperparameters can be modified, but by default, they are as follows:
Batch size: 128
Learning rate: 1e-4
Gamma: 0.99
Epsilon decay: 3000
Frame skip: 4, Frame stack: 4

init.lua (two variations: simple and complex)
The init.lua script is loaded into the Craftium Minetest environment (place here ~/Documents/craftium/craftium-envs/chop-tree/mods/craftium_mod/init.lua) 
and handles reward shaping and termination conditions. Included in the zip is init.lua and "complex init.lua". init.lua is ready to be used, while "complex init.lua"
contains all reward functions we used in the complex version of the model training.

Simple reward functions track the following:
When a tree block is dug (+15 reward)
When a non-tree is dug (-5 penalty)
If the dig key is held (+0.5 reward)
Complex reward functions also include the following:
Focused punches
Dig interruptions
Proximity to trees
Partial digs and stare time

Dependencies and how to run:
0. Set-up a virtual environment before following this guide to ensure nothing goes wrong.
1. First, use this to install dependences: "pip install numpy matplotlib torch gymnasium torchsummary"
2. Read the official docs for Craftium, and install following their instructions https://craftium.readthedocs.io/en/latest/installation/
3. Clone the environment directory for craftium, as linked above, and check for ~/craftium-envs/chop-tree to be certain you can train for tree chopping
4. Make sure the init.lua you want is placed here: ~/Documents/craftium/craftium-envs/chop-tree/mods/craftium_mod/init.lua
5. To run our code, open single.py and make sure train_agent(...) is uncommented and test_agent(...) is commented out. This will allow you to train the model.
6. When you have a model ready to go, comment out train_agent(...) and uncomment test_agent(...). This will run the agent.
7. Modify any Hyperparameters in single.py and reward functions in init.lua as needed.
8. You will be able to see the reward function plotting live, and observe the minecraft simulation visually. Note that during training, your mouse will not work.

Note:
The trained model will be saved as a .pth file you can re-use.
visual augmentations can be applied as needed.

References:

https://craftium.readthedocs.io/en/latest/installation/
https://stable-baselines3.readthedocs.io/en/master/guide/install.html
https://pypi.org/project/gymnasium/