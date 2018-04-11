import numpy as np
import matplotlib.pyplot as plt

frames = np.load('./frames.npy')
for frame in frames:
    plt.imshow(frame)
    plt.show()
