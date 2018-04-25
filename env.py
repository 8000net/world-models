import gym

import numpy as np
from scipy.misc import imresize

from keras.models import load_model

rnn = load_model('./models/mdn-rnn-forward.h5')
encoder = load_model('./models/encoder.h5')

def process_obs(obs):
    return np.expand_dims(imresize(obs, (64, 64)), axis=0) / 255.

# Controller set up
from keras.layers import Input, Dense
from keras.models import Model

z_dim = 32
lstm_units = 256
input_dim = z_dim + lstm_units

inputs = Input(shape=(input_dim,))
outputs = Dense(3)(inputs)

# TODO: this has `None` for gradient, need to set reward as gradient
def loss(reward, action):
    return reward

# X = z + h
# y = a
controller = Model(inputs, outputs)
controller.compile('adam', loss)

h = np.zeros(256)
rnn_out = np.zeros(256)

env = gym.make('CarRacing-v0')
obs = env.reset()
env.render()
obs = process_obs(obs)

done = False
cumulative_reward = 0
while not done:
    z = encoder.predict(obs)[0]

    # Choose action
    controller_in = np.expand_dims(np.hstack([z, h]), axis=0)
    a = controller.predict(controller_in)[0]
    print(a)

    obs, reward, done, info = env.step(a)
    env.render()
    obs = process_obs(obs)
    cumulative_reward += reward

    # Update controller's policy gradient
    # based on reward
    controller.fit(controller_in, np.array([reward]))

    # Forward rnn
    rnn_input = [
            np.array([[np.concatenate([z, a])]]),
            np.array([h]),
            np.array([rnn_out])
    ]

    h, rnn_out = rnn.predict(rnn_input)
    h = h[0]
    rnn_out = rnn_out[0]

