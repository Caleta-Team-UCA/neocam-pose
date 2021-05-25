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
        self.ser_right = ser
        self.ser_left = ser
        self.ser_up = ser
        self.ser_down = ser
        # Plot
        self.frequency = frequency
        self._timer = 0
        self._initialize_plot(size)

    def __len__(self):
        return len(self.ser_right)

    @property
    def ser_right_movavg(self) -> np.array:
        return moving_average(self.ser_right, size=self.frequency)

    @property
    def ser_left_movavg(self) -> np.array:
        return moving_average(self.ser_left, size=self.frequency)

    @property
    def ser_up_movavg(self) -> np.array:
        return moving_average(self.ser_up, size=self.frequency)

    @property
    def ser_down_movavg(self) -> np.array:
        return moving_average(self.ser_down, size=self.frequency)

    def _initialize_plot(self, size: int):
        """Initializes the plot"""
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        (self.line1,) = self.ax.plot(self.ser_right, label="Right")
        (self.line2,) = self.ax.plot(self.ser_left, label="Left")
        (self.line3,) = self.ax.plot(self.ser_up, label="Up")
        (self.line4,) = self.ax.plot(self.ser_down, label="Down")
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, 0.9)
        self.ax.legend()
        plt.show(block=False)

    def _update_plot(self):
        """Updates the plot lines"""
        self.line1.set_ydata(self.ser_right_movavg)
        self.line2.set_ydata(self.ser_left_movavg)
        self.line3.set_ydata(self.ser_up_movavg)
        self.line4.set_ydata(self.ser_down_movavg)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        body_detections: [dai.RawImgDetections],
        face_detections: [dai.RawImgDetections],
    ):
        """Updates the analysis with new information"""
        # Check if there are any body_detections
        try:
            # If there is a detection, compute the size of the box
            body = body_detections[0]
            face = face_detections[0]
            right = body.xmax - face.xmax
            left = face.xmin - body.xmin
            up = face.ymin - body.ymin
            down = body.ymax - face.ymax
        except (IndexError, TypeError) as er:
            # If not, the size is NaN
            right, left, up, down = np.nan, np.nan, np.nan, np.nan
        # Update the series of box sizes
        self.ser_right = np.append(self.ser_right[1:], right)
        self.ser_left = np.append(self.ser_left[1:], left)
        self.ser_up = np.append(self.ser_up[1:], up)
        self.ser_down = np.append(self.ser_down[1:], down)
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self._update_plot()
            self._timer = 0
