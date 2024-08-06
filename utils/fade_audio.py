import numpy as np


def get_factor(t: int, sr: int = 22400):
    n = int(t * sr)
    factors = np.arange(n) / n
    return factors
