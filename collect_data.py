import random

import numpy as np
import gym
from scipy.misc import imresize

# in rollouts
NUM_EPISODES = 2000
EPISODE_LEN = 300
FRAME_SKIP = 5
INIT_STEPS = 60
CHECKPOINT = 100

def get_action(t, a):
    if t < INIT_STEPS:
        return np.array([0, 1, 0])

    if t % FRAME_SKIP != 0:
        return a

    x = random.randint(0, 9)

    # Do nothing
    if x == 0:
        return np.array([0, 0, 0])

    # Accelerate
    if x in [1, 2, 3, 4]:
        return np.array([0, random.random(), 0])

    # Go left
    if x in [5, 6, 7]:
        return np.array([-random.random(), 0, 0])

    # Go right
    if x == 8:
        return np.array([random.random(), 0, 0])

    # Brake
    if x == 9:
        return np.array([0, 0, random.random()])


env = gym.make('CarRacing-v0')

frames = []
actions = []

i = 0
while i < NUM_EPISODES:
    print('Episode %d' % i)
    episode_frames = []
    episode_actions = []
    done = False
    obs = env.reset()
    env.render()
    t = 0
    a = None
    while t < EPISODE_LEN:
        a = get_action(t, a)

        episode_frames.append(imresize(obs, (64, 64)))
        episode_actions.append(a)
        obs, r, done, info = env.step(a)
        env.render()

        t += 1

    frames.append(episode_frames)
    actions.append(episode_actions)

    if i % CHECKPOINT == 0:
        np.save('frames.npy', frames)
        np.save('actions.npy', actions)

    i += 1

np.save('frames.npy', frames)
np.save('actions.npy', actions)
