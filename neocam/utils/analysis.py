import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class Analysis:
    fig: Figure
    ax: Axes
    line_height: Line2D
    line_width: Line2D

    def __init__(self, size: int = 1000, frequency: int = 10):
        # Initialize series
        ser = np.empty(size)
        ser[:] = np.nan
        self.ser_width = ser
        self.ser_height = ser
        # Plot
        self.frequency = frequency
        self._timer = 0
        self._initialize_plot(size)

    def __len__(self):
        return len(self.ser_width)

    def _initialize_plot(self, size: int):
        """Initializes the plot"""
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        (self.line_width,) = self.ax.plot(self.ser_width, label="Width")
        (self.line_height,) = self.ax.plot(self.ser_height, label="Height")
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, 1.5)
        self.ax.legend()
        plt.show(block=False)

    def _update_plot(self):
        """Updates the plot lines"""
        self.line_width.set_ydata(self.ser_width)
        self.line_height.set_ydata(self.ser_height)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, detections: list):
        """Updates the analysis with new information"""
        # Check if there are any detections
        try:
            # If there is a detection, compute the size of the box
            detection: dai.RawImgDetections = detections[0]
            x = detection.xmax - detection.xmin
            y = detection.ymax - detection.ymin
        except IndexError:
            # If not, the size is NaN
            x = np.nan
            y = np.nan
        # Update the series of box sizes
        self.ser_width = np.append(self.ser_width[1:], x)
        self.ser_height = np.append(self.ser_height[1:], y)
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self._update_plot()
            self._timer = 0
