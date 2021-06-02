import numpy as np
from matplotlib import pyplot as plt


class Dummy:
    coords = {
        "arm": {
            "close": {"x": np.array([0, 0.25, 0.5]), "y": np.array([7, 5, 7])},
            "open": {"x": np.array([0, 1, 1.5]), "y": np.array([7, 6.5, 8])},
        },
        "leg": {
            "close": {"x": np.array([0, 0.5, 1]), "y": np.array([4, 5, 3])},
            "open": {"x": np.array([0, 0.5, 0.5]), "y": np.array([4, 3, 1])},
        },
    }

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # Initialize body parts
        (self.body,) = self.ax.plot([0, 0, 0], [8, 7, 4], ".-", c="k")
        (self.arm_r,) = self.ax.plot(
            self.coords["arm"]["close"]["x"],
            self.coords["arm"]["close"]["y"],
            ".-",
            c="k",
        )
        (self.arm_l,) = self.ax.plot(
            -self.coords["arm"]["close"]["x"],
            self.coords["arm"]["close"]["y"],
            ".-",
            c="k",
        )
        (self.leg_r,) = self.ax.plot(
            self.coords["leg"]["close"]["x"],
            self.coords["leg"]["close"]["y"],
            ".-",
            c="k",
        )
        (self.leg_l,) = self.ax.plot(
            -self.coords["leg"]["close"]["x"],
            self.coords["leg"]["close"]["y"],
            ".-",
            c="k",
        )
        # Draw the figure
        self.fig.canvas.draw()
        self.ax.axis("off")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 10)
        plt.show(block=False)

    def open_right_leg(self):
        self.leg_r.set_xdata(self.coords["leg"]["open"]["x"])
        self.leg_r.set_ydata(self.coords["leg"]["open"]["y"])
        self.leg_r.set_color("r")

    def open_left_leg(self):
        self.leg_l.set_xdata(-self.coords["leg"]["open"]["x"])
        self.leg_l.set_ydata(self.coords["leg"]["open"]["y"])
        self.leg_l.set_color("r")

    def open_right_arm(self):
        self.arm_r.set_xdata(self.coords["arm"]["open"]["x"])
        self.arm_r.set_ydata(self.coords["arm"]["open"]["y"])
        self.arm_r.set_color("r")

    def open_left_arm(self):
        self.arm_l.set_xdata(-self.coords["arm"]["open"]["x"])
        self.arm_l.set_ydata(self.coords["arm"]["open"]["y"])
        self.arm_l.set_color("r")

    def close_right_leg(self):
        self.leg_r.set_xdata(self.coords["leg"]["close"]["x"])
        self.leg_r.set_ydata(self.coords["leg"]["close"]["y"])
        self.leg_r.set_color("k")

    def close_left_leg(self):
        self.leg_l.set_xdata(-self.coords["leg"]["close"]["x"])
        self.leg_l.set_ydata(self.coords["leg"]["close"]["y"])
        self.leg_l.set_color("k")

    def close_right_arm(self):
        self.arm_r.set_xdata(self.coords["arm"]["close"]["x"])
        self.arm_r.set_ydata(self.coords["arm"]["close"]["y"])
        self.arm_r.set_color("k")

    def close_left_arm(self):
        self.arm_l.set_xdata(-self.coords["arm"]["close"]["x"])
        self.arm_l.set_ydata(self.coords["arm"]["close"]["y"])
        self.arm_l.set_color("k")

    def update(self, down, right, left):
        if down < 3.1:
            self.close_right_leg()
            self.close_left_leg()
        else:
            self.open_right_leg()
            self.open_left_leg()
        if left < 0.6:
            self.close_left_arm()
        else:
            self.open_left_arm()
        if right < 0.6:
            self.close_right_arm()
        else:
            self.open_right_arm()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
