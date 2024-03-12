import gymnasium as gym
import matplotlib.pyplot as plt
import pprint as pprint

env = gym.make("MA-intersection-v0", render_mode='rgb_array')

pprint.pprint(env.unwrapped.config)

env.reset()

for _ in range(20):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()