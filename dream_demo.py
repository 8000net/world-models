import time
import random

import gym
import numpy as np
from scipy.misc import imresize, imsave
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rnn import loss, r_loss, kl_loss
from collect_data import get_action

z_dim = 32
a_dim = 3
input_dim = z_dim + a_dim
gaussian_mixtures = 5

def get_mixture_coef(output):
    d = gaussian_mixtures * z_dim

    pi = output[:,:d]
    mu = output[:,d:(2*d)]
    log_sigma = output[:,(2*d):(3*d)]

    # Pi put into softmax to ensure sum adds to one, and each
    # mixture probability is positive
    pi = np.exp(pi) / np.sum(np.exp(pi), axis=1, keepdims=True)

    sigma = np.exp(log_sigma)

    return pi, mu, sigma


def sample(pi, mu, sigma):
    # Select kth gaussian
    r = random.random()
    accum = 0
    for i, pi_k in enumerate(pi):
        if r < accum:
            break
        accum += pi_k

    k = int(i/160 * gaussian_mixtures)
    mu = mu[k*z_dim:(k+1)*z_dim]
    sigma = sigma[k*z_dim:(k+1)*z_dim]

    # Sample kth gaussian
    return np.random.normal(mu, sigma)

fig = plt.figure()


t = 0
z = np.load('./data/z-1.npy')[0][0]
a = np.load('./data/actions-1.npy')[0][0]
obs = None
rnn = load_model('./models/mdn-rnn.h5',
        custom_objects={'loss': loss, 'r_loss': r_loss, 'kl_loss': kl_loss})
decoder = load_model('./models/decoder.h5')
done = True

im = plt.imshow(np.zeros((64, 64, 3)), animated=True)

def update_fig(*args):
    global t, z, a, obs, rnn, decoder, done

    # Decode and display
    decoded = decoder.predict(np.array([z]))[0]
    im.set_array(decoded)

    # Sample dream frame for next iteration
    output = rnn.predict(np.array([[np.hstack([z, a])]]))[0]
    pi, mu, sigma = get_mixture_coef(output)
    pi = pi[0]
    mu = mu[0]
    sigma = sigma[0]
    z = sample(pi, mu, sigma)

    time.sleep(1/10)

    t += 1

    # Get random action for next state
    a = get_action(t, a)

    return im,




ani = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)
plt.show()
