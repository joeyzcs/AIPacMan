import gym


class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)

    def get_action(self, state):
        #  action = random.choice(range(self.action_size))
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action


while True:
    env = gym.make("CartPole-v1", render_mode='human')

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    agent = Agent(env)
    state = env.reset()


    for _ in range(200):
        #  action = env.action_space.sample()
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()


    env.close()