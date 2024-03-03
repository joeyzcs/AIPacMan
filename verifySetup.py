import gymnasium as gym

env = gym.make("ALE/Pong-v5", render_mode="human")
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    env.step(action)
env.close()
