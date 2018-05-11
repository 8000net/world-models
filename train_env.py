import multiprocessing

import gym

import numpy as np
from scipy.misc import imresize
from keras.models import load_model

from controller import Controller
from es import CMA_ES
from load_encoder import load_encoder

POP_SIZE = 12 # Number of solutions in each generation
STDDEV = 1.0
FITNESS_GOAL = 700
FRAME_SKIP = 5
NUM_WORKERS = POP_SIZE
N_ROLLOUTS_PER_TRIAL = 4

ENVS = [gym.make('CarRacing-v0') for _ in range(NUM_WORKERS)]

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.


def rollouts(agent, worker_i, n=N_ROLLOUTS_PER_TRIAL):
    env = ENVS[worker_i]
    encoder = load_encoder()
    #rnn = RNNS[worker_i]

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
                # turn off brake
                a[2] = 0
                #a = np.array([0, 1, 0])

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
        avg_reward += total_reward
        print('Worker %d: reward: %f' % (worker_i, total_reward))

    env.close()
    avg_reward /= n
    print('Worker %d: avg reward (over %d rollouts): %f' % (worker_i, n, avg_reward))
    return avg_reward


def start_work(worker_i, solution):
    print('Worker %d: started' % worker_i)
    controller = Controller(solution)
    return rollouts(controller, worker_i)


def train():
    phenotype = np.load('controller-params.npy')
    #phenotype = None
    solver = CMA_ES(phenotype=phenotype, pop_size=POP_SIZE,
                    n_dim=3*32+3, init_stddev=1.0)
    pool = multiprocessing.Pool(processes=NUM_WORKERS)

    gen = 0
    fitness_history = []
    while True:
        print('Generation %d' % gen)
        solutions = solver.ask()
        fitness_list = np.zeros(POP_SIZE)

        worker_results = [pool.apply_async(start_work, (i, solutions[i]))
                          for i in range(POP_SIZE)]
        fitness_list = [res.get() for res in worker_results]
        fitness_history.append(fitness_list)

        solver.tell(fitness_list)
        best_solution, best_fitness = solver.result()

        print('Generation %d: best fitness %f' % (gen, best_fitness))

        if best_fitness >= FITNESS_GOAL:
            np.save('controller-params.npy', best_solution)
            np.save('fitness-history.npy', np.array(fitness_history))
            break

        if gen > 1 and gen % 10 == 0:
            np.save('controller-params.npy', best_solution)
            np.save('fitness-history.npy', np.array(fitness_history))

        gen += 1


if __name__ == '__main__':
    print(NUM_WORKERS)
    train()
