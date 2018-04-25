import time
import random

import gym
import numpy as np
from scipy.misc import imresize, imsave
from keras.models import load_model

from gi.repository import Gtk
from threading import Thread

from rnn import loss, r_loss, kl_loss
from collect_data import get_action

z_dim = 32
a_dim = 3
input_dim = z_dim + a_dim
gaussian_mixtures = 5

IMAGE_PATH = '/tmp/wm.jpg'
class Window(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self)
        self.image = Gtk.Image()
        self.image.set_from_file(IMAGE_PATH)
        self.add(self.image)

    def refresh_image(self):
        self.image.set_from_file(IMAGE_PATH)


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

def save_decoded_z(z, decoder):
    z = np.expand_dims(z, axis=0)
    img = decoder.predict(z)[0]
    img = np.reshape(img, (64, 64, 3))
    imsave(IMAGE_PATH, img)

rnn = load_model('./models/mdn-rnn.h5',
        custom_objects={'loss': loss, 'r_loss': r_loss, 'kl_loss': kl_loss})
decoder = load_model('./models/decoder.h5')

z = np.load('./data/z-1.npy')[0][0]
a = np.load('./data/actions-1.npy')[0][0]

done = False
def stop(widget, data=None):
    global done
    done = True
    Gtk.main_quit()

win = Window()
win.connect("delete-event", stop)
save_decoded_z(z, decoder)
win.show_all()

window_thread = Thread(target=Gtk.main)
window_thread.start()

t = 0
while not done:
    a = get_action(t, a)
    output = rnn.predict(np.array([[np.hstack([z, a])]]))[0]
    pi, mu, sigma = get_mixture_coef(output)
    pi = pi[0]
    mu = mu[0]
    sigma = sigma[0]

    z = sample(pi, mu, sigma)
    save_decoded_z(z, decoder)
    win.refresh_image()

    t += 1
    time.sleep(1/10)
