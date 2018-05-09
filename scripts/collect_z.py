import glob

from keras.models import load_model
import numpy as np

encoder = load_model('./models/encoder.h5')

i = 1
batches = glob.glob('./data/frames-*.npy')
for batch in batches:
    print('Encoding %s...' % batch)
    zs = []
    frame_seqs = np.load(batch) / 255.
    for seq in frame_seqs:
        zs.append(encoder.predict(seq))
    np.save('./data/z-%d.npy' % i, zs)
    i += 1



# Check decoded output
#
# from scipy.misc import imsave
#
#i = 0
#for z in zs[0]:
#    z = np.expand_dims(z, axis=0)
#    img = decoder.predict(z)
#    img = np.reshape(img, (64, 64, 3))
#    imsave('./imgs/img-%d.jpg' % i, img*255)
#    i += 1
