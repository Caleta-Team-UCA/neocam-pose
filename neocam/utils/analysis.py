import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neocam.utils.series import Series


class Analysis:
    fig: Figure
    ax: Axes

    def __init__(self, size: int = 1000, frequency: int = 24):
        # Initialize series
        self.ser_right = Series(size=size, frequency=frequency, label="Right")
        self.ser_left = Series(size=size, frequency=frequency, label="Left")
        self.ser_up = Series(size=size, frequency=frequency, label="Up")
        self.ser_down = Series(size=size, frequency=frequency, label="Down")
        # Plot
        self.frequency = frequency
        self._timer = 0
        self._initialize_plot(size)

    def _initialize_plot(self, size: int):
        """Initializes the plot"""
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.ser_right.plot(self.ax)
        self.ser_left.plot(self.ax)
        self.ser_up.plot(self.ax)
        self.ser_down.plot(self.ax)
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, 0.9)
        self.ax.legend()
        plt.show(block=False)

    def _update_plot(self):
        """Updates the plot lines"""
        self.ser_right.update_plot()
        self.ser_left.update_plot()
        self.ser_up.update_plot()
        self.ser_down.update_plot()
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
        self.ser_right.append(right)
        self.ser_left.append(left)
        self.ser_up.append(up)
        self.ser_down.append(down)
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self._update_plot()
            self._timer = 0
