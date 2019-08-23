"""Neural network layer classes (only for online models)"""

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
        'relu': tf.keras.activations.relu,
        'linear': lambda x: x
    }

    def __init__(self, shape: tuple, name: str, activation: str = 'sigmoid'):
        """

        Args:
            shape (tuple): Output layer shape
            name (str): Name of the layer
            activation (str): Activation function of the layer. Initially it is a sigmoid
        """
        self._shape = shape
        self._name = name
        self._activation = self._activations[activation]

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

    def __init__(self, shape: tuple, name: str = 'input', activation: str = 'linear'):
        """
        Input layer initialization

        Args:
            shape (tuple): Shape of input neurons in the layer
            name (str): Name of the layer
            activation (str): Layer activation. Linear in the default
        """
        super().__init__(shape, name, activation)

    def __call__(self) -> tf.Tensor:
        return self._activation(tf.compat.v1.placeholder(tf.float32, self._shape, self._name))


class Dense(Layer):
    """Full connected layer"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', use_bias: bool = True, name: str = 'dense'):
        """
        Dense layer initialization.

        Args:
            shape (tuple): Shape of output layer tensor
            activation (str): Activation function of the layer. Initially it is a sigmoid
            use_bias (bool): If True use bias weights
            name (str): Name of the layer
        """
        super().__init__(shape, name, activation)
        self._use_bias = use_bias

    @name_scope
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Full connected layer.

        Args:
            tensor (tf.Tensor): Input tensor in the layer

        Return (tf.Tensor): Output tensor of the layer
        """
        tensor_shape = tuple(tensor.shape.as_list())
        axes0 = [-i for i in range(1, len(tensor_shape)+1)]
        axes1 = [i for i in range(len(tensor_shape))]

        weights = tf.Variable(tf.random.uniform(tensor_shape + self._shape, -1, 1, seed=self._seed, name='weights'))

        if self._use_bias:
            biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed, name='biases'))
            result = self._activation(tf.tensordot(tensor, weights, axes=[axes0, axes1]) + biases)
        else:
            result = self._activation(tf.tensordot(tensor, weights, axes=[axes0, axes1]))
        return result


class SRNN(Layer):
    """Simple recurrent neural network"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', name: str = 'srnn'):
        super().__init__(shape, name)
        self._activation = self._activations[activation]

    @name_scope
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        state = tf.Variable(tf.zeros(self._shape), trainable=False, name='state')
        state_weights = tf.Variable(tf.random.uniform(tuple(reversed(self._shape)) + self._shape, -1, 1,
                                                      seed=self._seed,
                                                      name='state_weights'))

        results = Dense(self._shape, activation='linear', use_bias=True, name='results')(tensor)

        axes0 = [-i for i in range(1, len(self._shape)+1)]
        axes1 = [i for i in range(len(self._shape))]

        for i in range(int(results.shape[0])):
            r = results[i]
            output = self._activation(r + tf.tensordot(state, state_weights, axes=[axes0, axes1]))
            state = tf.assign(state, output)
            results[i] = tf.assign(r, state)
        return results


# class LSTM(Layer):
#     """LSTM layer. Iterating through batch dimension and keep state between starting."""
#
#     def __init__(self, shape: tuple, state_shape: tuple = None, name: str = 'lstm'):
#         super().__init__(shape, name)
#         self._state_shape = state_shape if state_shape else shape
#
#     @name_scope
#     def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
#         state = tf.Variable(tf.zeros(self._state_shape), trainable=False, name='state')
#
#         with tf.name_scope('forget_gate'):
#             state_peephole = Dense(self._state_shape, activation='linear', name='state_peephole')(state)
#             input_peephole = Dense(self._state_shape, activation='linear', name='input_peephole')(tensor)
#             forget_gate = self._activations['sigmoid'](state_peephole + input_peephole)
#
#         with tf.name_scope('remember_gate'):
#             candidate_peephole = Dense(self._state_shape, activation='tanh', name='input_peephole')(tensor)
#             remember_gate = candidate_peephole * (1 - forget_gate)
#         forget_gate = cls.dense(inputs, num_outputs, name='forget_gate')
#         input_gate = cls.dense(inputs, num_outputs, name='input_gate')
#         candidate_gate = cls.dense(inputs, num_outputs, activation=tf.tanh, name='candidate_gate')
#
#         state = tf.assign(state, forget_gate * state + input_gate * candidate_gate)
#
#         output_gate = cls.dense(inputs, num_outputs, name='output_gate')
#
#         hidden = tf.assign(hidden, output_gate * tf.tanh(state))
#
#         return hidden


class Layer2(object):
    """Neural network layers"""

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
    inp = Input(shape=(3, 4, 5))
    model = Dense(shape=(6, 7, 8))(inp())
    model2 = SRNN(shape=(6, 7, 8))(inp())
