import json
from typing import List

import depthai as dai
import numpy as np

from neocam.pose.dummy import Dummy
from neocam.utils.plot_series import PlotSeries
from neocam.utils.series import Series


class Analysis:
    def __init__(
        self,
        size: int = 1000,
        frequency: int = 6,
    ):
        """Performs the pose analysis. Detects the move of each limb

        Parameters
        ----------
        size : int, optional
            Length of the stored series, by default 1000
        frequency : int, optional
            Rate at which plots are updated, in seconds, by default 6
        """
        # Initialize series
        self.ser_right = Series(size=size, frequency=frequency, label="Right")
        self.ser_left = Series(size=size, frequency=frequency, label="Left")
        self.ser_up = Series(size=size, frequency=frequency, label="Up")
        self.ser_down = Series(size=size, frequency=frequency, label="Down")

        self.frequency = frequency
        self._timer = 0
        # Plot series
        self.plot_series = PlotSeries(
            [self.ser_right, self.ser_left, self.ser_up, self.ser_down],
            xlim=(0, size),
            ylim=(0, 5),
        )
        # Plot dummy
        self.dummy = Dummy(self.ser_right, self.ser_left, self.ser_up, self.ser_down)

    @property
    def dict(self) -> dict:
        """Dictionary containing the series, in list format"""
        return {
            "right": self.ser_right.list,
            "left": self.ser_left.list,
            "up": self.ser_up.list,
            "down": self.ser_down.list,
        }

    def _update_series(
        self,
        body_detections: List[dai.RawImgDetections],
        face_detections: List[dai.RawImgDetections],
    ):
        """Updates the series with the last detections

        Parameters
        ----------
        body_detections : list
            List of body detections, in dai.RawImgDetections format
        face_detections : list
            List of face detections, in dai.RawImgDetections format
        """
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

    def update(
        self,
        body_detections: List[dai.RawImgDetections],
        face_detections: List[dai.RawImgDetections],
    ):
        """Updates the analysis with new information

        Parameters
        ----------
        body_detections : list
            List of body detections, in dai.RawImgDetections format
        face_detections : list
            List of face detections, in dai.RawImgDetections format
        """
        # Update the series
        self._update_series(body_detections, face_detections)
        # Update dummy
        self.dummy.update()

    def plot(self, frame: np.ndarray):
        """Updates the plots (series and dummy)

        Parameters
        ----------
        frame : numpy.ndarray
            Frame where the dummy is plotted
        """
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self.plot_series.update()
            self._timer = 0
        # Add dummy to baby image
        self.dummy.plot(frame)

    def to_json(self, path_json: str):
        """Stores the series in JSON format

        Parameters
        ----------
        path_json : str
            Output path to the json file
        """
        with open(path_json, "w") as outfile:
            json.dump(self.dict, outfile)
