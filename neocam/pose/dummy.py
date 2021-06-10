import numpy as np
import cv2


class Dummy:
    limbs = ["arm_left", "arm_right", "leg_left", "leg_right"]
    dict_coords = {
        "arm_right": {
            0: np.array([(0, 300), (25, 500), (50, 300)], dtype=np.int32),
            1: np.array([(0, 300), (100, 450), (150, 200)], dtype=np.int32),
        },
        "leg_right": {
            0: np.array([(0, 600), (50, 500), (100, 700)], dtype=np.int32),
            1: np.array([(0, 600), (50, 700), (50, 900)], dtype=np.int32),
        },
        "body": np.array([(0, 300), (0, 600)]),
    }
    status = [0, 0, 0, 0]  # arm left, right, leg left, right

    def __init__(self):
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
        self._move_dict_coords(self.dict_coords)

    def _move_dict_coords(self, dict_coords: dict, x: int = 300, y: int = 0):
        """Moves the coordinates contained in a dictionary"""
        for key, value in dict_coords.items():
            try:
                # Assume value is a numpy array and displace it
                dict_coords.update({key: value + np.array([x, y])})
            except TypeError:
                # If not, it is another dictionary. Nest the process
                self._move_dict_coords(value)

    def update(self, down: float, right: float, left: float):
        """Updates dummy status"""
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

    def plot(self, frame: np.ndarray):
        """Draws the dummy on a frame"""
        # Body
        cv2.polylines(
            frame,
            [self.dict_coords["body"]],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3,
        )
        # Limbs
        for idx, name in enumerate(self.limbs):
            cv2.polylines(
                frame,
                [self.dict_coords[name][self.status[idx]]],
                isClosed=False,
                color=(0, 255, 255 * self.status[idx]),
                thickness=3,
            )
