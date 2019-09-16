"""Module for different prepared neural network models"""
from typing import List, Tuple
import tensorflow as tf

from icscreator.kernel.layers import Input, SRNN, LSTM, VMLSTM
from icscreator.kernel.model import Model


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


class VMLSTMModel(Model):

    def __init__(self, input_shape: Tuple[int], memory_shape: Tuple[int], output_shape: Tuple[int], name: str = None):
        self._input_shape = input_shape
        self._memory_shape = memory_shape
        self._output_shape = output_shape
        super().__init__(name)

    def _build_model(self) -> (tf.Tensor, tf.Tensor):
        model_input = Input(shape=self._input_shape)
        model = VMLSTM(self._output_shape, self._memory_shape)(model_input)
        return model_input, model


if __name__ == "__main__":
    pass