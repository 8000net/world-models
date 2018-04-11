#TODO:
#   in CarRacing:
#   - fix frames only working if actually rendering
#   - episode ending early?

import numpy as np
import gym
from scipy.misc import imresize

# in rollouts
SAMPLE_SIZE = 1

env = gym.make('CarRacing-v0')

frames = []
actions = []

i = 0
while i < SAMPLE_SIZE:
    done = False
    obs = env.reset()
    while not done:
        frames.append(imresize(obs, (64, 64)))
        a = env.action_space.sample()
        actions.append(a)
        obs, r, done, info = env.step(a)

    i += 1

np.save('frames.npy', frames)
np.save('actions.npy', actions)
