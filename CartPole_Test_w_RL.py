import gym
import numpy as np
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Create the environment without new_step_api=True
env = gym.make("CartPole-v1")  # Removed new_step_api=True

states = env.observation_space.shape[0]
actions = env.action_space.n

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

print("States: ", states, " Actions: ", actions)

# Initialize the agent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

# Compile the agent
agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
# Train the agent - adjust nb_steps= to train agent further
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Test the agent
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

# Close the environment
env.close()


"""
RL practice packages - required versions

tensorflow 2.10
keras-rl2 1.0.5
gym 0.25.2

episodes = 10000
for episode in range(1, episodes+1):
    observation = env.reset()
    terminated = False
    truncated = False
    score = 0

    #while True:
    while not terminated and not truncated:
        action = random.choice([0, 1])
        observation, reward, terminated, truncated, info = env.step(action)  # Adjusted for new API
        score += reward
        if 'render_mode' in env.metadata and env.metadata['render_mode'] == 'human':
            env.render()

    print(f"Episode {episode}, Score: {score}")

env.close()
"""
