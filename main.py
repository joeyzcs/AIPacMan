import gymnasium as gym, time

env = gym.make("ALE/MsPacman-v5", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    #print(reward,"\n")
    #time.sleep(.5)

    if terminated or truncated:
        observation, info = env.reset()

env.close()