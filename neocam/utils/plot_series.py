from typing import List

import matplotlib.pyplot as plt

from neocam.utils.series import Series


class PlotSeries:
    def __init__(self, list_ser: List[Series], xlim: tuple = None, ylim: tuple = None):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.list_ser = list_ser
        for ser in self.list_ser:
            ser.plot(self.ax)
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.legend()
        plt.show(block=False)

    def update(self):
        for ser in self.list_ser:
            ser.update_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
