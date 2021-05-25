import numpy as np


def moving_average(ser: np.array, size: int = 24) -> np.array:
    """Computes the moving average of a series, returns a series of same length as the
    original (first elements remain the same)

    Parameters
    ----------
    ser : numpy.array
    size : int

    Returns
    -------
    numpy.array
    """
    ret = np.convolve(ser, np.ones(size), "valid") / size
    return np.append(ser[: (size - 1)], ret)
