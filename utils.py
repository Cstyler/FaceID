import numpy as np


class Scaler(object):
    def __init__(self, x=None, mean=None, std=None):
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(x, axis=0)
        if std is not None:
            self.std = std
        else:
            self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std




def transpose_matrix(x):
    return np.transpose(x, (0, 3, 1, 2))