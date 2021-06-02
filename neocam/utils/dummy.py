import numpy as np
from matplotlib import pyplot as plt


class Dummy:
    arm_close_x = np.array([0, 0.25, 0.5])
    arm_close_y = np.array([7, 5, 7])
    arm_open_x = np.array([0, 1, 1.5])
    arm_open_y = np.array([7, 6.5, 8])
    leg_close_x = np.array([0, 0.5, 1])
    leg_close_y = np.array([4, 5, 3])
    leg_open_x = np.array([0, 0.5, 0.5])
    leg_open_y = np.array([4, 3, 1])

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # Initialize body parts
        (self.body,) = self.ax.plot([0, 0, 0], [8, 7, 4], ".-", c="k")
        (self.arm_r,) = self.ax.plot(self.arm_close_x, self.arm_close_y, ".-", c="k")
        (self.arm_l,) = self.ax.plot(-self.arm_close_x, self.arm_close_y, ".-", c="k")
        (self.leg_r,) = self.ax.plot(self.leg_close_x, self.leg_close_y, ".-", c="k")
        (self.leg_l,) = self.ax.plot(-self.leg_close_x, self.leg_close_y, ".-", c="k")
        # Draw the figure
        self.fig.canvas.draw()
        self.ax.axis("off")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 10)
        plt.show(block=False)

    def open_right_leg(self):
        self.leg_r.set_xdata(self.leg_open_x)
        self.leg_r.set_ydata(self.leg_open_y)
        self.leg_r.set_color("r")

    def open_left_leg(self):
        self.leg_l.set_xdata(-self.leg_open_x)
        self.leg_l.set_ydata(self.leg_open_y)
        self.leg_l.set_color("r")

    def open_right_arm(self):
        self.arm_r.set_xdata(self.arm_open_x)
        self.arm_r.set_ydata(self.arm_open_y)
        self.arm_r.set_color("r")

    def open_left_arm(self):
        self.arm_l.set_xdata(-self.arm_open_x)
        self.arm_l.set_ydata(self.arm_open_y)
        self.arm_l.set_color("r")

    def close_right_leg(self):
        self.leg_r.set_xdata(self.leg_close_x)
        self.leg_r.set_ydata(self.leg_close_y)
        self.leg_r.set_color("k")

    def close_left_leg(self):
        self.leg_l.set_xdata(-self.leg_close_x)
        self.leg_l.set_ydata(self.leg_close_y)
        self.leg_l.set_color("k")

    def close_right_arm(self):
        self.arm_r.set_xdata(self.arm_close_x)
        self.arm_r.set_ydata(self.arm_close_y)
        self.arm_r.set_color("k")

    def close_left_arm(self):
        self.arm_l.set_xdata(-self.arm_close_x)
        self.arm_l.set_ydata(self.arm_close_y)
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
