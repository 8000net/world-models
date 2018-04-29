import multiprocessing

import gym

import numpy as np
from scipy.misc import imresize
from keras.models import load_model

from controller import Controller
from es import ES

POP_SIZE = 6 # Number of solutions in each generation
STDDEV = 1.0
FITNESS_GOAL = 100
FRAME_SKIP = 5
NUM_WORKERS = POP_SIZE
N_ROLLOUTS_PER_TRIAL = 4
ENVS = [gym.make('CarRacing-v0') for _ in range(NUM_WORKERS)]

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.


def rollouts(agent, env_i, n=N_ROLLOUTS_PER_TRIAL):
    env = ENVS[env_i]
    rnn = load_model('./models/mdn-rnn-forward.h5')
    encoder = load_model('./models/encoder.h5')

    avg_reward = 0
    for i in range(n):
        #h = np.zeros(256)
        #rnn_out = np.zeros(256)

        obs = env.reset()
        env.render()
        obs = process_obs(obs)
        a = np.array([0, 1, 0])

        done = False
        total_reward = 0
        t = 1
        while not done:
            z = encoder.predict(obs)[0]

            # Choose action
            if t % FRAME_SKIP == 0:
                #a = agent.get_action(np.hstack([z, h]))
                a = agent.get_action(z)

            obs, reward, done, info = env.step(a)
            env.render()
            obs = process_obs(obs)
            total_reward += reward
            t += 1

            # Update rnn and get new h
            #rnn_input = [
            #        np.array([[np.concatenate([z, a])]]),
            #        np.array([h]),
            #        np.array([rnn_out])
            #]
            #h, rnn_out = rnn.predict(rnn_input)
            #h = h[0]
            #rnn_out = rnn_out[0]

    env.close()
    avg_reward /= n
    print('Avg reward (over %d rollouts): %d' % (n, avg_reward))
    return avg_reward


def start_work(env, solution):
    controller = Controller(solution)
    return rollouts(controller, env)


solver = ES(pop_size=POP_SIZE, n_dim=3*32+3, stddev=1.0)
pool = multiprocessing.Pool(processes=NUM_WORKERS)

gen = 0
while True:
    print('Generation %d' % gen)
    solutions = solver.ask()
    fitness_list = np.zeros(POP_SIZE)

    worker_results = [pool.apply_async(start_work, (i, solutions[i]))
                      for i in range(POP_SIZE)]
    fitness_list = [res.get() for res in worker_results]

    solver.tell(fitness_list)
    best_solution, best_fitness = solver.result()

    print('Generation %d: chose best solution with fitness %f' % (
        gen, best_fitness))

    if best_fitness >= FITNESS_GOAL:
        np.save('controller-params.npy', best_solution)
        break

    gen += 1
