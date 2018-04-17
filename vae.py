import numpy as np

from keras.layers import (
        Input, Conv2D, Conv2DTranspose, Lambda, Dense, Flatten, Reshape)
from keras.models import Model
import keras.backend as K

input_dim = 64, 64, 3
latent_dim = 32
EPOCHS = 1
BATCH_SIZE = 1

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
decoder_h1 = Dense(1024, name='decoder_h1')
decoder_h2 = Reshape((1, 1, 1024), name='decoder_reshape')
decoder_h3 = Conv2DTranspose(128, 5, strides=2, activation='relu', name='decoder_h3')
decoder_h4 = Conv2DTranspose(64, 5, strides=2, activation='relu', name='decoder_h4')
decoder_h5 = Conv2DTranspose(32, 6, strides=2, activation='relu', name='decoder_h5')
decoder_outputs = Conv2DTranspose(3, 6, strides=2, activation='sigmoid', name='decoder_out')

# VAE Decoder
h = decoder_h1(z)
h = decoder_h2(h)
h = decoder_h3(h)
h = decoder_h4(h)
h = decoder_h5(h)
outputs = decoder_outputs(h)

# Decoder
_z = Input(shape=(latent_dim,))
_h = decoder_h1(_z)
_h = decoder_h2(_h)
_h = decoder_h3(_h)
_h = decoder_h4(_h)
_h = decoder_h5(_h)
_outputs = decoder_outputs(_h)

l2_loss = K.sum(K.square(inputs - outputs)) / 2
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                       axis=-1)

vae = Model(inputs, outputs)
vae_loss = K.mean(l2_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=None)

encoder = Model(inputs, z)
decoder = Model(_z, _outputs)

frames_path = 'frames.npy'
frames = np.load(frames_path)
n_episodes, n_frames, w, h, c = frames.shape
frames = np.reshape(frames, (n_episodes * n_frames, w, h, c)) / 255.

vae.fit(frames, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE)

vae.save('vae.h5')
encoder.save('encoder.h5')
decoder.save('decoder.h5')
