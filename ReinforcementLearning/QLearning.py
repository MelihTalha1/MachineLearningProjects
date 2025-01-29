import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

action = environment.action_space.sample()
"""
left: 0
down: 1
right: 2
up: 3
"""

# S1 -> (Action 1) -> S2
new_state, reward, done, info, _ = environment.step(action)


    







