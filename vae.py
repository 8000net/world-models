from keras.layers import (
        Input, Conv2D, Conv2DTranspose, Lambda, Dense, Flatten, Reshape)
from keras.models import Model
import keras.backend as K

input_dim = 64, 64, 3
latent_dim = 32
epochs = 1

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
        mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

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

# Decoder layers
h = Dense(1024)(z)                                                   # 1024
h = Reshape((1, 1, 1024))(h)                                         # 1x1x1024
h = Conv2DTranspose(128, 5, strides=2, activation='relu')(h)         # 5x5x128
h = Conv2DTranspose(64, 5, strides=2, activation='relu')(h)          # 13x13x64
h = Conv2DTranspose(32, 6, strides=2, activation='relu')(h)          # 30x30x32
outputs = Conv2DTranspose(3, 6, strides=2, activation='sigmoid')(h)  # 64x64x3

# Place holder for decoder
decoder_input = Input(shape=(latent_dim,))


l2_loss = K.sum(K.square(inputs - outputs)) / 2
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                       axis=-1)

encoder = Model(inputs, z)

vae = Model(inputs, outputs)
vae_loss = K.mean(l2_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss=None)
