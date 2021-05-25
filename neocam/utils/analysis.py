import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from neocam.utils.series import moving_average


class Analysis:
    fig: Figure
    ax: Axes
    line1: Line2D
    line2: Line2D

    def __init__(self, size: int = 1000, frequency: int = 24):
        # Initialize series
        ser = np.empty(size)
        ser[:] = np.nan
        self.ser_ratio = ser
        # Plot
        self.frequency = frequency
        self._timer = 0
        self._initialize_plot(size)

    def __len__(self):
        return len(self.ser_ratio)

    @property
    def ser_ratio_movavg(self) -> np.array:
        return moving_average(self.ser_ratio, size=self.frequency)

    def _initialize_plot(self, size: int):
        """Initializes the plot"""
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        (self.line1,) = self.ax.plot(self.ser_ratio, label="Raw")
        (self.line2,) = self.ax.plot(self.ser_ratio_movavg, label="Moving average")
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, 1.5)
        self.ax.legend()
        plt.show(block=False)

    def _update_plot(self):
        """Updates the plot lines"""
        self.line1.set_ydata(self.ser_ratio)
        self.line2.set_ydata(self.ser_ratio_movavg)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, detections: [dai.RawImgDetections]):
        """Updates the analysis with new information"""
        # Check if there are any detections
        try:
            # If there is a detection, compute the size of the box
            detection = detections[0]
            x = detection.xmax - detection.xmin
            y = detection.ymax - detection.ymin
        except (IndexError, TypeError) as er:
            # If not, the size is NaN
            x = np.nan
            y = np.nan
        # Update the series of box sizes
        self.ser_ratio = np.append(self.ser_ratio[1:], x / y)
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self._update_plot()
            self._timer = 0
