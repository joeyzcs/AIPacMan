import gymnasium
import time
import numpy as np

env = gymnasium.make("ALE/MsPacman-ram-v5", render_mode="human")  # Initialize MsPacman environment
env.action_space.seed(42)

observation, info = env.reset(seed=42)  # Reset to initial state

print(env.observation_space)
print(env.observation_space.sample())

"""
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())  #  Update environment (action)

    #print(observation,"\n")
    #time.sleep(.2)

    if terminated or truncated:  #  Reach end of state
        observation, info = env.reset()

"""
env.close()
