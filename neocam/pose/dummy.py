import numpy as np
import cv2

from neocam.utils.series import Series


class Dummy:
    width: float = 1920
    height: int = 1080
    limbs = ["arm_left", "arm_right", "leg_left", "leg_right"]
    dict_coords = {
        "arm_right": {
            0: np.array([(0, 100), (25, 300), (50, 150)], dtype=np.int32),
            1: np.array([(0, 100), (100, 250), (150, 100)], dtype=np.int32),
        },
        "leg_right": {
            0: np.array([(0, 400), (50, 300), (100, 500)], dtype=np.int32),
            1: np.array([(0, 400), (50, 500), (50, 700)], dtype=np.int32),
        },
        "body": np.array([(0, 100), (0, 400)], dtype=np.int32),
    }
    status = [0, 0, 0, 0]  # arm left, right, leg left, right

    def __init__(
        self,
        ser_right: Series,
        ser_left: Series,
        ser_up: Series,
        ser_down: Series,
        width: int = 1920,
        height: int = 1080,
    ):
        """Plots a dummy inside a frame, portraying the newborns limbs' positions

        Parameters
        ----------
        ser_right : neocam.utils.series.Series
        ser_left : neocam.utils.series.Series
        ser_up : neocam.utils.series.Series
        ser_down : neocam.utils.series.Series
        """
        # Store series
        self.ser_right = ser_right
        self.ser_left = ser_left
        self.ser_up = ser_up
        self.ser_down = ser_down
        # shape (1080, 1920, 3)
        self.dict_coords.update(
            {
                "arm_left": {
                    0: self.dict_coords["arm_right"][0] * np.array([-1, 1]),
                    1: self.dict_coords["arm_right"][1] * np.array([-1, 1]),
                },
                "leg_left": {
                    0: self.dict_coords["leg_right"][0] * np.array([-1, 1]),
                    1: self.dict_coords["leg_right"][1] * np.array([-1, 1]),
                },
            }
        )
        # Displace the coordinates accordingly
        self.dict_coords = self._move_dict_coords(self.dict_coords, 300, 300)
        self.resize(width, height)

    def _move_dict_coords(self, dict_coords: dict, x: int, y: int):
        """Moves the coordinates contained in a dictionary

        Parameters
        ----------
        dict_coords : dict
            Dictionary containing the dummy coordinates
        x : int
            Displacement on X axis
        y : int
            Displacement on Y axis
        Returns
        -------
        dict
            New dictionary of dummy coordinates, displaced
        """
        # Do not modify original dict
        dict_copy = dict_coords.copy()
        for key, value in dict_coords.items():
            try:
                # Assume value is a numpy array and displace it
                new_value = np.array(value + np.array([x, y]), dtype=np.int32)
            except TypeError:
                # If not, it is another dictionary. Nest the process
                new_value = self._move_dict_coords(value, x, y)
            # Update dictionary
            dict_copy.update({key: new_value})
        return dict_copy

    def _resize_dict_coords(self, dict_coords: dict, width: int, height: int):
        """Resizes the coordinates contained in a dictionary

        Parameters
        ----------
        dict_coords : dict
            Dictionary containing the dummy coordinates
        width : int, optional
            Image width
        height : int, optional
            Image height
        Returns
        -------
        dict
            New dictionary of dummy coordinates, resized
        """
        # Do not modify original dict
        dict_copy = dict_coords.copy()
        for key, value in dict_coords.items():
            try:
                # Assume value is a numpy array and displace it
                new_value = np.array(
                    value * np.array([width / self.width, height / self.height]),
                    dtype=np.int32,
                )
            except TypeError:
                # If not, it is another dictionary. Nest the process
                new_value = self._resize_dict_coords(value, width, height)
            # Update dictionary
            dict_copy.update({key: new_value})
        return dict_copy

    def resize(self, width: int, height: int):
        """Resizes the dummy according to new image size

        Parameters
        ----------
        width : int
        height : int
        """
        self.dict_coords = self._resize_dict_coords(self.dict_coords, width, height)
        self.width, self.height = width, height

    def update(self):
        """Updates dummy status"""
        right = self.ser_right.movavg[-1]
        left = self.ser_left.movavg[-1]
        down = self.ser_down.movavg[-1]
        if down < 3.1:
            self.status[2] = 0
            self.status[3] = 0
        else:
            self.status[2] = 1
            self.status[3] = 1
        if left < 0.6:
            self.status[0] = 0
        else:
            self.status[0] = 1
        if right < 0.6:
            self.status[1] = 0
        else:
            self.status[1] = 1

    def plot(self, frame: np.ndarray) -> np.ndarray:
        """Draws the dummy on a frame

        Parameters
        ----------
        frame : numpy.ndarray
            Frame where the dummy is drawn

        Returns
        -------
        numpy.ndarray
            Frame with dummy plotted
        """
        # Copy the original array
        frame_new = frame.copy()
        # Body
        cv2.polylines(
            frame_new,
            [self.dict_coords["body"]],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3,
        )
        # Limbs
        for idx, name in enumerate(self.limbs):
            cv2.polylines(
                frame_new,
                [self.dict_coords[name][self.status[idx]]],
                isClosed=False,
                color=(0, 255, 255 * self.status[idx]),
                thickness=3,
            )
        return frame_new
