import gymnasium as gym
from matplotlib import pyplot as plt
import pprint

env = gym.make('highway-v0', render_mode='rgb_array')
env.unwrapped.config["lanes_count"] = 12
env.unwrapped.config["vehicles_density"] = 0.5
env.unwrapped.config["screen_height"] = 450
pprint.pprint(env.unwrapped.config)
env.reset()
for _ in range(10):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.reset()
plt.imshow(env.render())
plt.show()

print(obs, reward)