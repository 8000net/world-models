import gym

import numpy as np
from scipy.misc import imresize
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collect_data import get_action

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.

encoder = load_model('./models/encoder.h5')
decoder = load_model('./models/decoder.h5')
env = gym.make('CarRacing-v0')

fig = plt.figure()

done = True
t = 0
a = None
obs = None

im = plt.imshow(np.zeros((64, 64, 3)), animated=True)

def update_fig(*args):
    global t, a, env, obs, encoder, decoder, done

    if done == True:
        t = 0
        obs = env.reset()
        env.render()
        obs = process_obs(obs)

    z = encoder.predict(obs)
    decoded = decoder.predict(z)[0]

    a = get_action(t, a)
    obs, reward, done, info = env.step(a)
    env.render()
    obs = process_obs(obs)
    t += 1

    im.set_array(decoded)
    return im,

ani = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)
plt.show()
