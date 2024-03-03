import gymnasium as gym

# Correctly list all registered environments
for env_id in gym.envs.registry.keys():
    print(env_id)

# Correctly check specifically for ALE environments
ale_envs = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("ALE/")]
if not ale_envs:
    print("No ALE environments are registered.")
else:
    for env_id in ale_envs:
        print(env_id)

