"""Module for different prepared neural network models"""

from icscreator.kernel.layers import Input, Dense, SRNN
import tensorflow as tf
from typing import List
from abc import abstractmethod
import numpy as np


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
        self._model = self._build_model()
        self._input_shape = self._model.input_shape
        self._output_shape = self._model.output_shape
        self._optimizer = None
        self._loss = None
        self._session = tf.keras.backend.get_session()

    @abstractmethod
    def _build_model(self) -> tf.keras.Model:
        raise NotImplemented("Method '_build_model' is not implemented is {}".format(self.__class__.__name__))

    def set_training_parameters(self, loss: str, optimizer: str, optimizer_parameters: dict):
        """

        :param loss_function: https://www.tensorflow.org/api_docs/python/tf/losses
        :param optimizer: https://www.tensorflow.org/api_docs/python/tf/train
        :param optimizer_parameters:
            adam: {
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08}
            gradient: {
                learning_rate}
            momentum: {
                learning_rate,
                momentum,
                use_nesterov=False}
            rms: {
                learning_rate,
                decay=0.9,
                momentum=0.0,
                epsilon=1e-10}
        :return:
        """
        assert optimizer in self._OPTIMIZERS, "Use optimizers from the list: {}".format(list(self._OPTIMIZERS.keys()))
        assert loss in self._LOSSES, "Use losses from the list: {}".format(list(self._LOSSES.keys()))
        self._loss = self._LOSSES[loss]
        self._optimizer = self._OPTIMIZERS[optimizer](**optimizer_parameters)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result = []
        input_tensor = self._model.input
        output_tensor = self._model.output
        for data in input_data:
            r = self._session.run(output_tensor, feed_dict={input_tensor: data})
            result.append(r)
        result = np.array(result)
        return result

    def update_weights(self, loss: float):
        gradients = tf.keras.backend.gradients(loss, self._model.trainable_variables)
        # with tf.GradientTape() as tape:
        #     gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    def fit(self, input_data, target_data, epochs: int = 1000, batch: int = 10, repetitions: int = 100):
        self._session.as_default()
        for e in range(epochs):
            for r in range(repetitions):
                inp = input_data[r*batch:(r+1)*batch]
                tar = target_data[r*batch:(r+1)*batch]
                model_result = self.predict(inp)
                loss = self._loss(tar, model_result)
                # loss = 0.1
                self.update_weights(loss)
                print("Epoch {}: loss = {}".format(e, loss))


class MultilayerSRNN(Model):

    def __init__(self, layer_sizes: List[int], activations: List[str] = ('sigmoid',), name: str = None):
        self._layer_sizes = layer_sizes
        self._activations = activations * (len(layer_sizes) - 1) if len(activations) == 1 else activations
        super().__init__(name)

    def _build_model(self) -> tf.keras.Model:
        model_input = Input(shape=(self._layer_sizes[0],))
        model = SRNN(shape=(self._layer_sizes[1],), activation=self._activations[0])(model_input)
        for layer, activation in zip(self._layer_sizes[2:], self._activations[1:]):
            model = SRNN(shape=(layer,), activation=activation)(model)
        model = tf.keras.Model(model_input, model, name=self._name)
        return model


def multilayer_perceptron(*layer_sizes, activations: List[str] = ('sigmoid',)) -> tf.keras.Model:
    assert len(layer_sizes) > 1, "Network should have more than one layer"
    if len(activations) == 1:
        activations = activations * (len(layer_sizes) - 1)
    else:
        assert len(layer_sizes) - 1 == len(activations)

    model_input = Input(layer_sizes[0])
    model = Dense(shape=layer_sizes[1], activation=activations[0], name='dense_0')(model_input)
    for e, (layer, activation) in enumerate(zip(layer_sizes[2:], activations[1:])):
        model = Dense(shape=layer, activation=activation, name='dense_{}'.format(e+1))(model)
    model = tf.keras.Model(model_input, model)
    return model


def elman_network(inputs, hiddens, outputs, activation: str = 'sigmoid') -> tf.keras.Model:
    inputs = (inputs,) if isinstance(inputs, int) else inputs
    hiddens = (hiddens,) if isinstance(hiddens, int) else hiddens
    outputs = (outputs,) if isinstance(outputs, int) else outputs

    model_input = Input(shape=inputs)
    model = SRNN(output_shape=hiddens, activation='relu')(model_input)
    model = Dense(shape=outputs, activation=activation)(model)
    model = tf.keras.Model(model_input, model)
    return model


if __name__ == "__main__":
    input_data = np.random.uniform(-100, 100, (10000, 5))
    target_data = np.random.uniform(-100, 100, (10000, 3))

    model = MultilayerSRNN(layer_sizes=[5, 10, 15, 3])
    print(model.predict(input_data[:10]))
    model.set_training_parameters('mse', 'adam', {'learning_rate': 0.001})
    model.fit(input_data, target_data)
