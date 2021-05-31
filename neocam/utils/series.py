import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


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
    return np.append(np.repeat(np.nan, size - 1), ret)


class Series:
    line: Line2D = None

    def __init__(self, size: int = 1000, frequency: int = 24, label: str = ""):
        """Series

        Parameters
        ----------
        size : int, optional
            Length of the series, by default 1000
        frequency : int, optional
            Size of the moving average, by default 24
        label : str, optional
            Name of the series, by default ""
        """
        self.ser = np.empty(size)
        self.ser[:] = np.nan
        self.frequency = frequency
        self.label = label

    def __len__(self):
        return len(self.ser)

    def append(self, x: float):
        self.ser = np.append(self.ser[1:], x)

    @property
    def movavg(self):
        return moving_average(self.ser, size=self.frequency)

    def plot(self, ax: Axes):
        """Plots the line of given axes"""
        (self.line,) = ax.plot(self.ser, label=self.label)

    def update_plot(self):
        """Updates the plotted line"""
        self.line.set_ydata(self.movavg)
