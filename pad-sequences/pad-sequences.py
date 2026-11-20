import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    # Handling the empty array
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Finding the maximum length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    #Creating the array filled with the pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copying the sequences with padding or truncation
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result
    pass