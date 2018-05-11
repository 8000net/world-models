import numpy as np

from keras.layers import (
        Input, Conv2D, Lambda, Dense, Flatten)
from keras.models import Model
import keras.backend as K

input_dim = 64, 64, 3
latent_dim = 32
EPOCHS = 1
BATCH_SIZE = 32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 32),
        mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def load_encoder():
    # Encoder layers
    inputs = Input(shape=input_dim)                                      # 64x64x3
    h = Conv2D(32, 4, strides=2, activation='relu')(inputs)              # 31x31x32
    h = Conv2D(64, 4, strides=2, activation='relu')(h)                   # 14x14x64
    h = Conv2D(128, 4, strides=2, activation='relu')(h)                  # 6x6x128
    h = Conv2D(256, 4, strides=2, activation='relu')(h)                  # 2x2x256
    h = Flatten()(h)                                                     # 1024
    z_mean = Dense(latent_dim, name='z_mean')(h)                         # 32
    z_log_var = Dense(latent_dim, name='z_log_var')(h)                   # 32
    z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

    encoder = Model(inputs, z)
    encoder.load_weights('models/encoder.h5')

    return encoder
