"""Visualization tool"""

import math
import numpy as np

from typing import List
from matplotlib import pyplot as plt


class VisualTool(object):
    """Visualization of training process"""

    def __init__(self, titles: List[str], x_info: List[str], y_info: List[str], legend: List[str],
                 ratio: float = 16 / 9, show_loss: bool = True):
        """
        Initialization of visualization tool

        Args:
            titles ([str]): List of titles for each neural network output
            x_info ([str]): List of x axes info for each neural network output
            y_info ([str]): List of y axes info for each neural network output
            legend ([str]): List of plot legends
            ratio (float): Ratio of subplot tiles x_tiles / y_tiles. Might be your monitor aspect ratio
            show_loss (bool): Flag for showing loss values during training process
        """
        assert len(titles) == len(x_info) == len(y_info)
        self._titles = titles
        self._x_info = x_info
        self._y_info = y_info
        self._legend = legend
        self._ratio = ratio
        self._show_loss = show_loss

        self._rows, self._cols = self._get_tiles_ratio()
        self._fig = plt.figure()
        self._construct_tiles()

    def _get_tiles_ratio(self):
        num_plots = len(self._titles)
        if self._ratio < 1:
            cols = math.floor(math.pow(num_plots, self._ratio))
            rows = math.ceil(num_plots / cols)
        else:
            rows = math.floor(math.pow(num_plots, 1 / self._ratio))
            cols = math.ceil(num_plots / rows)
        return rows, cols

    def _construct_tiles(self):
        if self._show_loss:
            gridsize = (self._rows, 2 * self._cols)
            self._loss_ax = plt.subplot2grid(gridsize, (0, 0), colspan=self._cols, rowspan=self._rows)
            self._add_markup_to_loss()
            grid_coords = [(y, x) for y in range(self._rows) for x in range(2 * self._cols) if x >= self._cols]
        else:
            gridsize = (self._rows, self._cols)
            grid_coords = [(y, x) for y in range(self._rows) for x in range(self._cols)]

        self._param_axs = [plt.subplot2grid(gridsize, gc) for gc in grid_coords[:len(self._titles)]]
        self._add_markup_to_params()
        plt.show()

    def _add_markup_to_loss(self):
        self._loss_ax.set_title("Losses")
        self._loss_ax.set_xlabel("Epochs")
        self._loss_ax.set_ylabel("Loss values")
        self._loss_ax.grid(True)
        self._loss_ax.legend(self._titles, loc="upper left")

    def _add_markup_to_params(self):
        for i, ax in enumerate(self._param_axs):
            ax.set_title(self._titles[i])
            ax.set_xlabel(self._x_info[i])
            ax.set_ylabel(self._y_info[i])
            ax.grid(True)
            ax.legend(self._legend)

    def draw(self, losses: List[float], data_to_draw: List[np.ndarray]):
        assert len(data_to_draw) == len(self._legend)
        for i in data_to_draw:
            assert len(i.shape) == 2
            assert i.shape[0] == len(self._titles)

        for sbplt in self._axs:
            pass


if __name__ == "__main__":
    num_param = 2
    num_data = 10
    ratio = 9 / 16

    titles = ["Param_{}".format(n) for n in range(num_param)]
    x_info = ["X axis for param_{}".format(n) for n in range(num_param)]
    y_info = ["Y axis for param_{}".format(n) for n in range(num_param)]
    legend = ["Data_{}".format(n) for n in range(num_data)]

    losses = [list(1 / np.random.uniform(0, 5, 100).cumsum()) for _ in range(num_data)]

    vis = VisualTool(titles, x_info, y_info, legend, ratio, show_loss=True)
    rows, cols = vis._get_tiles_ratio()
    print((rows, cols))

    for i in range(len(losses[0])):
        param_losses = [losses[n][i] for n in range(len(losses))]
        data = [np.random.uniform(-100, 100, (100, num_param)) for _ in range(num_data)]

