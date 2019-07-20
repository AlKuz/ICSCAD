"""
Recurrent neural networks:

ElmanNetwork
"""
import numpy as np
import os
import tensorflow as tf
from abc import abstractmethod


class Layer(object):
    """Neural network layers"""

    seed = None

    @classmethod
    def input(cls, num_inputs: int, name='input') -> tf.Tensor:
        """
        Input layer.

        Args:
            num_inputs (int): Number of input neurons in the layer
            name (str): Name of the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        return tf.compat.v1.placeholder(tf.float32, [num_inputs], name)

    @classmethod
    def state(cls, num_outputs: int, name='state') -> tf.Variable:
        """
        Untrainable variable for keeping state inside neural network.

        Args:
            num_outputs (int): Number of output neurons in the layer
            name (str): Name of the layer

        Return (tf.Variable): Tensorflow variable
        """
        return tf.Variable(tf.zeros([num_outputs]), trainable=False, name=name)

    @classmethod
    def assign(cls, target, value) -> tf.Tensor:
        return tf.assign(target, value)

    @classmethod
    def dense(cls, tensor: tf.Tensor, num_outputs: int, activation=tf.sigmoid, name='dense') -> tf.Tensor:
        """
        Full connected layer.

        Args:
            tensor (tf.Tensor): Input tensor in the layer
            num_outputs (int): Number of output neurons in the layer
            activation (func): Activation function of the layer. Initially it is a sigmoid
            name (str): Name of the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        assert len(tensor.shape) == 1
        num_inputs = int(tensor.shape[0])

        with tf.name_scope(name):
            biases = tf.Variable(tf.random.uniform([num_outputs], -1, 1, seed=cls.seed, name='biases'))
            weights = tf.Variable(tf.random.uniform([num_outputs, num_inputs], -1, 1, seed=cls.seed, name='weights'))
            result = activation(tf.linalg.matvec(weights, tensor) + biases)

        return result

    @classmethod
    def lstm(cls, tensor: tf.Tensor, num_outputs: int, name='lstm') -> tf.Tensor:
        """
        Long-short term memory layer.

        Args:
            tensor (tf.Tensor): Input tensor in the layer
            num_outputs (int): Number of output neurons in the layer
            name (str): Name of the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        with tf.name_scope(name):
            state = tf.Variable(tf.zeros([num_outputs]), trainable=False, name='state')
            hidden = tf.Variable(tf.zeros([num_outputs]), trainable=False, name='hidden')

            inputs = tf.concat([hidden, tensor], axis=0)

            forget_gate = cls.dense(inputs, num_outputs, name='forget_gate')
            input_gate = cls.dense(inputs, num_outputs, name='input_gate')
            candidate_gate = cls.dense(inputs, num_outputs, activation=tf.tanh, name='candidate_gate')

            state = tf.assign(state, forget_gate * state + input_gate * candidate_gate)

            output_gate = cls.dense(inputs, num_outputs, name='output_gate')

            hidden = tf.assign(hidden, output_gate * tf.tanh(state))

            return hidden

    @classmethod
    def delay(cls, tensor: tf.Tensor, num_delays: int, name='delay') -> tf.Tensor:
        """
        Layer for delay input tensor in multiple steps.

        Args:
            tensor (tf.Tensor): Input tensor in the layer
            num_delays (int): Number of delays
            name (str): Name of the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        assert len(tensor.shape) == 1
        num_inputs = int(tensor.shape[0])

        with tf.name_scope(name):
            delays = tf.Variable(tf.zeros([num_inputs * (num_delays + 1)]), trainable=False, name='tensor_delays')
            concatenated = tf.concat([tensor, delays[:-num_inputs]], axis=0)
            delays = tf.assign(delays, concatenated)
        return delays


class NeuralNetwork(object):
    """Base class of neural networks"""

    def __init__(self,
                 loss=tf.losses.mean_squared_error,
                 optimizer=tf.train.AdamOptimizer,
                 optimizer_settings={},
                 name='neural_network',
                 model_folder=os.getcwd()):
        """
        Initialization of base network class.
        """
        self._name = name
        self._model_folder = model_folder
        self._session = tf.Session()
        self._model_inputs = None
        self._model_outputs = None
        self._model_targets = None
        self._model_loss = None
        self._model_optimizer = None

        self.compile(loss, optimizer, **optimizer_settings)

    def compile(self,
                loss=tf.losses.mean_squared_error,
                optimizer=tf.train.AdamOptimizer,
                **optimizer_settings) -> None:

        self._session.as_default()
        with tf.name_scope(self._name):
            self._create_model()

        assert isinstance(self._model_inputs, tf.Tensor)
        assert isinstance(self._model_outputs, tf.Tensor)
        self._model_targets = Layer.input(self._model_outputs.shape[1], name='targets')

        self._model_loss = loss(self._model_targets, self._model_outputs)
        self._model_optimizer = optimizer(**optimizer_settings).minimize(self._model_loss)
        self._session.run(tf.initialize_all_variables())

    @abstractmethod
    def _create_model(self):
        """
        Method for model creating.

        Use:
            self._model_inputs for input placeholder
            self._model_outputs for output tensor
        """
        pass

    def predict(self, data):
        """
        Make prediction of neural network.

        Args:
            data (list): Make predictions for list of values.

        Returns (list): List of prediction results.
        """
        result = []
        for i in data:
            r = self._session.run(self._model_outputs, feed_dict={self._model_inputs: i})
            r = list(r) if len(r) > 1 else float(r)
            result.append(r)
        return result

    def evaluate(self, input_data: list, target_data: list) -> float:
        """
        Evaluate neural network.

        Args:
            input_data (list): List of data to feed neural network.
            target_data (list): List of data to compare network results.

        Returns (float):
        """
        loss_result = []
        for i, t in zip(input_data, target_data):
            r = self._session.run(self._model_loss, feed_dict={self._model_inputs: i,
                                                               self._model_targets: t})
            loss_result.append(r)
        loss_result = float(np.mean(loss_result))
        return loss_result

    def optimize(self, error: float):
        pass

    def train(self, input_data: list, target_data: list):
        pass

    def save(self, path: str, only_model=False):
        pass


if __name__ == "__main__":
    class ElmanNetwork(NeuralNetwork):
        """Elman network"""

        def __init__(self, inputs: int, hiddens: int, outputs: int, seed=None):
            self._inputs = inputs
            self._hiddens = hiddens
            self._outputs = outputs
            Layer.seed = seed
            super().__init__()

        def _create_model(self):
            self._model_inputs = Layer.input(self._inputs)
            state = Layer.state(self._hiddens)
            hiddens = Layer.dense(tf.concat([self._model_inputs, state], axis=0), self._hiddens, name='hidden')
            hiddens = Layer.assign(target=state, value=hiddens)
            self._model_outputs = Layer.dense(hiddens, self._outputs, name='output')

        @property
        def hiddens(self):
            return self._hiddens

        @property
        def inputs(self):
            return self._inputs

        @property
        def outputs(self):
            return self._outputs

    data = [[1, 2, 3]] * 10
    model1 = ElmanNetwork(3, 10, 2, seed=13)
    model2 = ElmanNetwork(3, 10, 1, seed=13)
    print(model1.predict(data[:2]))
    print(model2.predict(data[:2]))
