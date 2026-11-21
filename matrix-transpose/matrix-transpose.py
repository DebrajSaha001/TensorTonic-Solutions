import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    # Convert input into NumPy array
    A = np.asarray(A)

    # Get dimensions
    M, N = A.shape

    # Create output matrix with reversed dimensions
    result = np.zeros((N, M), dtype=A.dtype)

    # Swap row and column indices
    for i in range(M):
        for j in range(N):
            result[j][i] = A[i][j]

    return result
