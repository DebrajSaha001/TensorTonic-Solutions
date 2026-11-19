import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    # Getting the number of samples and features
    N, D = X.shape

    # Initializing the weights and bias
    w = np.zeros(D)
    b = 0.0

    # Gradient Descent
    for _ in range(steps):
        # Forward propagation
        z = X @ w + b

        # Apply sigmoid
        p = _sigmoid(z)

        # Calculate error
        error = p - y

        # Calculate gradients
        dw = (X.T @ error) / N
        db = np.mean(error)

        # Update weights and bias
        w -= lr * dw
        b -= lr * db

    return w, float(b)
    # pass