"""
Visualization tool

The Tensorboard realization was taken from
https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e
"""
import io
import math
import numpy as np
import tensorflow as tf
from tensorboard import main as tb

from PIL import Image
from typing import List
from matplotlib import pyplot as plt


class Tensorboard(object):

    def __init__(self, log_folder: str):
        self._writer = tf.summary.FileWriter(log_folder)
        tf.flags.FLAGS.logdir = log_folder
        tb.run_main()

    def __del__(self):
        self._writer.close()

    def log_scalar(self, tag, value, global_step: int):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_histogram(self, tag, values, global_step: int, bins: int):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_image(self, tag, img, global_step: int):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_plot(self, tag, figure, global_step: int):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                       height=img_ar.shape[0],
                                       width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

class EmptyVisualTool(object):

    def draw(self, data_to_draw: List[List[np.ndarray]], losses: List[float] = None):
        pass


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

        plt.ion()
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

    def _add_markup_to_loss(self):
        self._loss_ax.set_title("Accuracy")
        self._loss_ax.set_xlabel("Epochs")
        self._loss_ax.set_ylabel("Errors, %")
        self._loss_ax.grid(True)
        self._loss_ax.legend(self._titles, loc="upper left")

    def _add_markup_to_params(self):
        for i, ax in enumerate(self._param_axs):
            ax.set_title(self._titles[i])
            ax.set_xlabel(self._x_info[i])
            ax.set_ylabel(self._y_info[i])
            ax.grid(True)
            ax.legend(self._legend, loc="upper left")

    def draw(self, data_to_draw: List[List[np.ndarray]], accuracy: List[float] = None):
        if self._show_loss and accuracy is not None:
            try:
                self._accuracy: np.ndarray = np.append(self._accuracy, np.array([accuracy]), axis=0)
            except AttributeError:
                self._accuracy = np.array([accuracy])
            self._draw_ax(self._loss_ax, self._accuracy, graph_type='log')
            self._add_markup_to_loss()
            plt.pause(0.0001)

        prepared_data = np.transpose(np.array(data_to_draw), axes=(2, 1, 0))
        for i, ax in enumerate(self._param_axs):
            self._draw_ax(ax, prepared_data[i, ...])
        self._add_markup_to_params()

    def _draw_ax(self, ax, data: np.ndarray, graph_type='plot'):
        ax.clear()
        if graph_type == 'plot':
            ax.plot(data)
        elif graph_type == 'log':
            ax.semilogy(data)

    def show(self):
        plt.pause(100500)

    def __del__(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    num_param = 2
    num_data = 2
    ratio = 9 / 16

    titles = ["Param_{}".format(n) for n in range(num_param)]
    x_info = ["X axis for param_{}".format(n) for n in range(num_param)]
    y_info = ["Y axis for param_{}".format(n) for n in range(num_param)]
    legend = ["Data_{}".format(n) for n in range(num_data)]

    losses = [list(1 / np.random.uniform(0, 5, 100).cumsum()) for _ in range(num_data)]

    vis = VisualTool(titles, x_info, y_info, legend, ratio, show_loss=True)

    for i in range(len(losses[0])):
        param_losses = [losses[n][i] for n in range(len(losses))]
        data = [np.random.uniform(-100, 100, (10000, num_param)) for _ in range(num_data)]
        vis.draw(data, param_losses)
