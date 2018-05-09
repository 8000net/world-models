import gym

import numpy as np
from scipy.misc import imresize
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from controller import Controller

VIEW_DECODED = False
BRAKE_DAMPEN = 1

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.

encoder = load_model('./models/encoder.h5')
decoder = load_model('./models/decoder.h5')
rnn = load_model('./models/mdn-rnn-forward.h5')
controller_params = np.load('./models/controller-params.npy')
controller = Controller(controller_params)
env = gym.make('CarRacing-v0')


done = True
t = 0
a = None
obs = None

if VIEW_DECODED:
    fig = plt.figure()
    im = plt.imshow(np.zeros((64, 64, 3)), animated=True)

    def update_fig(*args):
        global t, a, env, obs, encoder, decoder, done

        if done == True:
            t = 0
            obs = env.reset()
            env.render()
            obs = process_obs(obs)

        z = encoder.predict(obs)[0]
        decoded = decoder.predict(np.array([z]))[0]

        a = controller.get_action(z)
        obs, reward, done, info = env.step(a)
        env.render()
        obs = process_obs(obs)
        t += 1

        im.set_array(decoded)
        return im,

    ani = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)
    plt.show()

else:
    while True:
        if done == True:
            t = 0
            obs = env.reset()
            env.render()
            obs = process_obs(obs)

        z = encoder.predict(obs)[0]
        decoded = decoder.predict(np.array([z]))[0]

        a = controller.get_action(z)
        a *= np.array([1, 1, 1-BRAKE_DAMPEN])
        obs, reward, done, info = env.step(a)
        env.render()
        obs = process_obs(obs)
        t += 1
