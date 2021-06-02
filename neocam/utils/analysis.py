import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neocam.utils.dummy import Dummy
from neocam.utils.series import Series


class Analysis:
    fig: Figure
    ax: Axes

    def __init__(
        self,
        size: int = 1000,
        frequency: int = 24,
        plot_series: bool = False,
        dummy: bool = True,
    ):
        # Initialize series
        self.ser_right = Series(size=size, frequency=frequency, label="Right")
        self.ser_left = Series(size=size, frequency=frequency, label="Left")
        self.ser_up = Series(size=size, frequency=frequency, label="Up")
        self.ser_down = Series(size=size, frequency=frequency, label="Down")

        self.frequency = frequency
        self._timer = 0
        # Plot series
        self.plot_series = plot_series
        self._initialize_plot(size)
        # Plot dummy
        self.plot_dummy = dummy
        self._initialize_dummy()

    def _initialize_plot(self, size: int):
        """Initializes the plot"""
        if not self.plot_series:
            return
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
        self.ax.set_ylim(0, 5)
        self.ax.legend()
        plt.show(block=False)

    def _initialize_dummy(self):
        if not self.plot_dummy:
            return
        self.dummy = Dummy()

    def _update_series(
        self,
        body_detections: [dai.RawImgDetections],
        face_detections: [dai.RawImgDetections],
    ):
        """Updates the series"""
        # Check if there are any body_detections
        try:
            # If there is a detection, compute the size of the box
            body = body_detections[0]
            face = face_detections[0]
            # Compute the face size
            size = face.ymax - face.ymin
            right = (body.xmax - face.xmax) / size
            left = (face.xmin - body.xmin) / size
            up = (face.ymin - body.ymin) / size
            down = (body.ymax - face.ymax) / size
        except (IndexError, TypeError) as er:
            # If not, the size is the last value
            right = self.ser_right.ser[-1]
            left = self.ser_left.ser[-1]
            up = self.ser_up.ser[-1]
            down = self.ser_down.ser[-1]
        # Update the series of box sizes
        self.ser_right.append(right)
        self.ser_left.append(left)
        self.ser_up.append(up)
        self.ser_down.append(down)

    def _update_plot(self):
        """Updates the plot lines"""
        if not self.plot_series:
            return
        self.ser_right.update_plot()
        self.ser_left.update_plot()
        self.ser_up.update_plot()
        self.ser_down.update_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_dummy(self):
        down = self.ser_down.movavg[-1]
        left = self.ser_left.movavg[-1]
        right = self.ser_right.movavg[-1]
        self.dummy.update(down, right, left)

    def _plot_dummy(self, frame: np.ndarray):
        if not self.plot_dummy:
            return
        self.dummy.plot(frame)

    def update(
        self,
        body_detections: [dai.RawImgDetections],
        face_detections: [dai.RawImgDetections],
    ):
        """Updates the analysis with new information"""
        # Update the series
        self._update_series(body_detections, face_detections)
        # Update dummy
        self._update_dummy()

    def plot(self, frame: np.ndarray):
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self._update_plot()
            self._timer = 0
        # Add dummy to baby image
        self._plot_dummy(frame)
