import numpy as np
import cv2


class Dummy:
    limbs = ["arm_left", "arm_right", "leg_left", "leg_right"]
    coords = {
        "arm_left": {
            0: np.array([(0, 700), (25, 500), (50, 700)], dtype=np.int32),
            1: np.array([(0, 700), (100, 650), (150, 800)], dtype=np.int32),
        },
        "arm_right": {
            0: np.array([(0, 700), (25, 500), (50, 700)], dtype=np.int32),
            1: np.array([(0, 700), (100, 650), (150, 800)], dtype=np.int32),
        },
        "leg_left": {
            0: np.array([(0, 400), (50, 500), (100, 300)], dtype=np.int32),
            1: np.array([(0, 400), (50, 300), (50, 100)], dtype=np.int32),
        },
        "leg_right": {
            0: np.array([(0, 400), (50, 500), (100, 300)], dtype=np.int32),
            1: np.array([(0, 400), (50, 300), (50, 100)], dtype=np.int32),
        },
    }

    status = [0, 0, 0, 0]  # arm left, right, leg left, right

    def __init__(self):
        pass

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
        print(frame.shape)
        for idx, name in enumerate(self.limbs):
            cv2.polylines(
                frame,
                [self.coords[name][self.status[idx]]],
                isClosed=False,
                color=(0, 255, 255 * self.status[idx]),
                thickness=3,
            )
