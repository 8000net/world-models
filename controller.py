import math

import numpy as np

class Controller:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_action(self, x):
        a = np.dot(self.W, x) + self.b
        a = self._scale_output(a)
        return a

    def _scale_output(self, output):
        output[0] = np.tanh(output[0])
        output[1] = (np.tanh(output[1]) + 1) / 2
        output[2] = (np.tanh(output[2]) + 1) / 2
        return output
