import numpy as np
import cv2


class Dummy:
    limbs = ["arm_left", "arm_right", "leg_left", "leg_right"]
    coords = {
        "arm_right": {
            0: np.array([(0, 300), (25, 500), (50, 300)], dtype=np.int32),
            1: np.array([(0, 300), (100, 450), (150, 200)], dtype=np.int32),
        },
        "leg_right": {
            0: np.array([(0, 600), (50, 500), (100, 700)], dtype=np.int32),
            1: np.array([(0, 600), (50, 700), (50, 900)], dtype=np.int32),
        },
    }

    status = [0, 0, 0, 0]  # arm left, right, leg left, right

    def __init__(self):
        # shape (1080, 1920, 3)
        self.coords.update(
            {
                "arm_left": {
                    0: self.coords["arm_right"][0] * np.array([-1, 1]),
                    1: self.coords["arm_right"][1] * np.array([-1, 1]),
                },
                "leg_left": {
                    0: self.coords["leg_right"][0] * np.array([-1, 1]),
                    1: self.coords["leg_right"][1] * np.array([-1, 1]),
                },
            }
        )
        # Move coords to right
        for limb in self.limbs:
            for status in [0, 1]:
                self.coords[limb].update(
                    {status: self.coords[limb][status] + np.array([300, 0])}
                )

    def update(self, down, right, left):
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
        for idx, name in enumerate(self.limbs):
            cv2.polylines(
                frame,
                [self.coords[name][self.status[idx]]],
                isClosed=False,
                color=(0, 255, 255 * self.status[idx]),
                thickness=3,
            )
