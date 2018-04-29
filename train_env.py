import gym

import numpy as np
from scipy.misc import imresize
from keras.models import load_model

from controller import Controller

NUM_SOLUTIONS = 5 # Number of solutions in each generation
STDDEV = 1.0
FITNESS_GOAL = 100

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.


def rollout(agent, env, rnn, encoder):
    h = np.zeros(256)
    rnn_out = np.zeros(256)

    obs = env.reset()
    env.render()
    obs = process_obs(obs)

    done = False
    total_reward = 0
    while not done:
        z = encoder.predict(obs)[0]

        # Choose action
        a = agent.get_action(np.hstack([z, h]))

        obs, reward, done, info = env.step(a)
        env.render()
        obs = process_obs(obs)
        total_reward += reward

        # Forward rnn
        rnn_input = [
                np.array([[np.concatenate([z, a])]]),
                np.array([h]),
                np.array([rnn_out])
        ]
        h, rnn_out = rnn.predict(rnn_input)
        h = h[0]
        rnn_out = rnn_out[0]

    print('Total reward: %d' % total_reward)
    return total_reward


rnn = load_model('./models/mdn-rnn-forward.h5')
encoder = load_model('./models/encoder.h5')
env = gym.make('CarRacing-v0')


# Simple Evolution Strategy
i = 0
mean = np.zeros(867)
cov = np.diag(np.ones(867) * STDDEV)
print(mean.shape)
while True:
    print('Generation %d' % i)
    solutions = [np.random.multivariate_normal(mean, cov)
                 for _ in range(NUM_SOLUTIONS)]

    fitness_list = np.zeros(NUM_SOLUTIONS)

    for i, solution in enumerate(solutions):
        W = np.reshape(solution[:864], (3, 288))
        b = solution[864:]
        controller = Controller(W, b)
        fitness_list[i] = rollout(controller, env, rnn, encoder)

    max_i = np.argmax(fitness_list)
    mean = solutions[max_i]
    print('Generation %d: chose best solution with fitness %f' % (
        i, fitness_list[max_i]))

    if fitness_list[max_i] >= FITNESS_GOAL:
        break
