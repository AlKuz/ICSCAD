"""Neural network layer class"""

import tensorflow as tf
from typing import List
from abc import abstractmethod


def name_scope(method):
    def wrapper(self, *args):
        with tf.name_scope(self._name):
            return method(self, *args)
    return wrapper


class Layer(object):
    """Base layer class"""

    _seed = None
    _activations = {
        'sigmoid': tf.keras.activations.sigmoid,
        'tanh': tf.keras.activations.tanh,
        'relu': tf.keras.activations.relu
    }

    def __init__(self, shape: tuple, name: str):
        self._shape = shape
        self._name = name

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tf.Tensor:
        raise Exception("Method doesn't implemented")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed: int):
        self._seed = new_seed


class Input(Layer):
    """Input layer"""

    def __init__(self, shape: tuple, name: str = 'input'):
        """
        Input layer initialization

        Args:
            shape (tuple): Shape of input neurons in the layer
            name (str): Name of the layer
        """
        super().__init__((None,) + shape, name)

    def __call__(self) -> tf.Tensor:
        return tf.compat.v1.placeholder(tf.float32, self._shape, self._name)


class Dense(Layer):
    """Full connected layer"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', name: str = 'dense'):
        """
        Dense layer initialization.

        Cases:
            Dense(shape=(6,))(tf.placeholder((None, 3, 4, 5))) -> (None, 3, 4, 6)
            Dense(shape=(6, 7))(tf.placeholder((None, 3, 4, 5))) -> (None, 3, 6, 7)
            Dense(shape=(6, 7, 8))(tf.placeholder((None, 3, 4, 5))) -> (None, 6, 7, 8)
            Dense(shape=(6, 7, 8, 9))(tf.placeholder((None, 3, 4, 5))) -> (None, 6, 7, 8, 9)

        Args:
            shape (tuple): Shape of input neurons in the layer
            activation (str): Activation function of the layer. Initially it is a sigmoid
            name (str): Name of the layer
        """
        super().__init__(shape, name)
        self._activation = self._activations[activation]

    @name_scope
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Full connected layer.

        Args:
            tensor (tf.Tensor): Input tensor in the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        tensor_shape = tuple(map(lambda x: int(x), tensor.shape[1:]))[-len(self._shape):]
        if isinstance(tensor_shape, int):
            tensor_shape = (tensor_shape,)
        biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed, name='biases'))
        weights = tf.Variable(tf.random.uniform(tensor_shape + self._shape, -1, 1, seed=self._seed, name='weights'))
        axes0 = [-i for i in range(1, len(tensor_shape)+1)]
        axes1 = [i for i in range(len(tensor_shape))]
        result = self._activation(tf.tensordot(tensor, weights, axes=[axes0, axes1]) + biases)
        return result


class Layer2(object):
    """Neural network layers"""

    seed = None

    @classmethod
    def input(cls, shape: tuple, name='input') -> tf.Tensor:
        """
        Input layer.

        Args:
            shape (tuple): Shape of input neurons in the layer
            name (str): Name of the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        assert len(shape) > 1, "Shape must contain batch in the first dimension"
        return tf.compat.v1.placeholder(tf.float32, shape, name)

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
    def concat(cls, tensors_list: List[tf.Tensor], axis: int = 0, name: str = 'concat'):
        return tf.concat(tensors_list, axis, name)

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


if __name__ == "__main__":
    d = Dense(shape=(6, 7))(tf.compat.v1.placeholder(tf.float32, (None, 3, 4, 5)))
