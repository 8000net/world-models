import math

import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
import keras.backend as K

z_dim = 32
a_dim = 3
input_dim = z_dim + a_dim
lstm_units = 256
gaussian_mixtures = 5
mdn_units = gaussian_mixtures * 3 * z_dim # phi, mu, sigma
EPOCHS = 20
BATCH_SIZE = 32

def get_mixture_coef(output):
    d = gaussian_mixtures * z_dim

    seq_length = K.shape(output)[1]

    pi = output[:,:,:d]
    mu = output[:,:,d:(2*d)]
    log_sigma = output[:,:,(2*d):(3*d)]

    pi = K.reshape(pi, [-1, seq_length, gaussian_mixtures, z_dim])
    mu = K.reshape(mu, [-1, seq_length, gaussian_mixtures, z_dim])
    log_sigma = K.reshape(log_sigma, [-1, seq_length, gaussian_mixtures, z_dim])

    # Pi put into softmax to ensure sum adds to one, and each
    # mixture probability is positive
    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)

    sigma = K.exp(log_sigma)

    return pi, mu, sigma


oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
def pdf(y, mu, sigma, pi):
    seq_length = K.shape(y)[1]
    y = K.tile(y, (1, 1, gaussian_mixtures))
    y = K.reshape(y, [-1, seq_length, gaussian_mixtures, z_dim])

    # Calculate pdfs of z in individiual gaussians
    pdfs = oneDivSqrtTwoPI*(1/sigma)*K.exp(-K.square(y-mu)/(2*K.square(sigma)))

    # Take weighted sum of pdfs
    return K.sum(pi*pdfs, axis=2)


def r_loss(y, output):
    """
    Reconstruction loss

    Log likelihood of generated prob dist "explaining" y
    """
    pi, mu, sigma = get_mixture_coef(output)

    return -K.mean(K.log(pdf(y, mu, sigma, pi)), axis=(1, 2))


def kl_loss(y, output):
    """
    Kullback-Leibler divergence loss term

    Measure difference between the distribution of z, to an IID gaussian
    vector with mean = 0, var = 1
    """
    pi, mu, sigma = get_mixture_coef(output)
    return -0.5*K.mean(1+K.log(sigma)-K.square(mu)-sigma, axis=[1, 2, 3])

def loss(y, output):
    return r_loss(y, output) + kl_loss(y, output)


# Training model
inputs = Input(shape=(None, input_dim))
lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
lstm_out, _, _ = lstm(inputs)
outputs = Dense(mdn_units, name='rnn_mdn_out')(lstm_out)

rnn = Model(inputs, outputs)

# Prediction model
# inputs_h - hidden state
# inputs_c - last output from LSTM
#
# Agent should keep the last hidden state and output,
# and feed into this model to get the next hidden state

inputs_h = Input(shape=(lstm_units,))
inputs_c = Input(shape=(lstm_units,))
_, state_h, state_c = lstm(inputs, initial_state=[inputs_h, inputs_c])

forward = Model([inputs, inputs_h, inputs_c], [state_h, state_c])


rnn.compile('adam', loss)

for i in range(1, 2):
    print('Loading batch %d...' % i)
    z = np.load('./data/z-%i.npy' % i)
    actions = np.load('./data/actions-%i.npy' % i)
    X = []
    Y = []
    for seq_z, seq_a in zip(z, actions):
        seq_za = []

        for frame_z, frame_a in zip(seq_z, seq_a):
            seq_za.append(np.hstack([frame_z, frame_a]))

        # Store x_i as z_i + a_i, and y_i as z_i+1
        X.append(seq_za[:-1])
        Y.append(seq_z[1:])

    X = np.array(X)
    Y = np.array(Y)

    rnn.fit(X, Y, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.2)

rnn.save('mdn-rnn.h5')
forward.save('mdn-rnn-forward.h5')
