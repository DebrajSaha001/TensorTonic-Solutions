import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    if np.isscalar(x):
        return float(1 / (1 + np.exp(-x)))

    x = np.asarray(x, dtype=float)

    return 1 / (1 + np.exp(-x))
