"""Module for different prepared neural network models"""

from icscreator.kernel.layers import Input, Dense, SRNN, LSTM
import tensorflow as tf
from typing import List, Tuple
from abc import abstractmethod
import numpy as np
from tqdm import tqdm
from icscreator.kernel.visualization import VisualTool, EmptyVisualTool


class Model(object):

    _OPTIMIZERS = {
        'adam': tf.train.AdamOptimizer,
        'gradient': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'rms': tf.train.RMSPropOptimizer
    }
    _LOSSES = {
        'mse': tf.losses.mean_squared_error,
        'sigmoid_cross_entropy': tf.losses.sigmoid_cross_entropy,
        'softmax_cross_entropy': tf.losses.softmax_cross_entropy
    }

    def __init__(self, name: str = None):
        self._name = self.__class__.__name__ if name is None else name
        self._session = tf.Session()
        self._input = None
        self._output = None
        self._target = None
        self._optimizer = None
        self._loss = None

    @abstractmethod
    def _build_model(self) -> (tf.Tensor, tf.Tensor):
        """
        Create custom model graph

        Returns:
            (tf.Tensor, tf.Tensor): (model_input, model_output)
        """
        raise NotImplemented("Method '_build_model' is not implemented is {}".format(self.__class__.__name__))

    def compile(self, loss: str, optimizer: str, optimizer_parameters: dict):
        """
        Compile model graph

        Args:
            loss (str): Cost function (https://www.tensorflow.org/api_docs/python/tf/losses)
            optimizer (str): Optimizer to update model weights (https://www.tensorflow.org/api_docs/python/tf/train)
            optimizer_parameters (dict):
                adam: {learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08}
                gradient: {learning_rate}
                momentum: {learning_rate, momentum, use_nesterov=False}
                rms: {learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10}
        """
        assert optimizer in self._OPTIMIZERS, "Use optimizers from the list: {}".format(list(self._OPTIMIZERS.keys()))
        assert loss in self._LOSSES, "Use losses from the list: {}".format(list(self._LOSSES.keys()))
        self._session.as_default()
        self._input, self._output = self._build_model()
        self._target = Input(shape=self._output.shape.as_list(), name='Target')
        self._loss = self._LOSSES[loss](self._target, self._output)
        self._optimizer = self._OPTIMIZERS[optimizer](**optimizer_parameters).minimize(self._loss)
        self._session.run(tf.compat.v1.global_variables_initializer())

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result = []
        for data in input_data:
            r = self._session.run(self._output, feed_dict={self._input: data})
            result.append(r)
        result = np.array(result)
        return result

    def fit(self, input_data, target_data, epochs: int = 1000, vizualizer: VisualTool = EmptyVisualTool):
        self._session.as_default()
        mean_loss = None
        for e in range(epochs):
            losses = []
            for input_sample, target_sample in tqdm(zip(input_data, target_data),
                                                    desc="Epoch {}: loss = {}".format(e, mean_loss)):
                loss, _ = self._session.run([self._loss, self._optimizer], feed_dict={
                    self._input: input_sample,
                    self._target: target_sample}
                )
                losses.append(loss)
            mean_loss = sum(losses) / len(losses)
            model_result = self.predict(input_data)
            losses = np.std(target_data - model_result, axis=0)
            vizualizer.draw([model_result, target_data], list(losses))


class MultilayerSRNN(Model):

    def __init__(self, layer_sizes: List[int], activations: List[str] = ('sigmoid',), name: str = None):
        self._layer_sizes = layer_sizes
        self._activations = activations * (len(layer_sizes) - 1) if len(activations) == 1 else activations
        super().__init__(name)

    def _build_model(self) -> (tf.Tensor, tf.Tensor):
        model_input = Input(shape=(self._layer_sizes[0],))
        model = SRNN(shape=(self._layer_sizes[1],), activation=self._activations[0])(model_input)
        for layer, activation in zip(self._layer_sizes[2:], self._activations[1:]):
            model = SRNN(shape=(layer,), activation=activation)(model)
        return model_input, model


class LSTMModel(Model):

    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int], name: str = None):
        self._input_shape = input_shape
        self._output_shape = output_shape
        super().__init__(name)

    def _build_model(self) -> (tf.Tensor, tf.Tensor):
        model_input = Input(shape=self._input_shape)
        model = LSTM(self._output_shape)(model_input)
        return model_input, model


if __name__ == "__main__":
    DATA = "/home/alexander/Projects/ICSCreator/static/data/Data_JC.csv"

    data = np.genfromtxt(DATA, delimiter=',', skip_header=True)
    fuel = np.expand_dims(data[::100, 1], axis=-1) / 4.0
    freq = np.expand_dims(data[::100, 2], axis=-1) / 200000.0
    temp = np.expand_dims(data[::100, 3], axis=-1) / 1000.0
    output = np.concatenate([freq, temp], axis=-1)

    vis_tool = VisualTool(
        titles=['Rotor frequency', 'Turbine temperature'],
        x_info=['Time step'] * 2,
        y_info=['Normalized frequency', 'Normalized temperature'],
        legend=['Model', 'Target'],
        ratio=1/2,
        show_loss=True
    )

    network_model = MultilayerSRNN(layer_sizes=[1, 30, 2])
    network_model.compile('mse', 'adam', {'learning_rate': 0.001})
    print(network_model.predict(fuel[:10]))
    network_model.fit(fuel, output, epochs=1000, vizualizer=vis_tool)
