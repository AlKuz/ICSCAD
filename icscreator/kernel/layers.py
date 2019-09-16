"""Neural network layer classes (only for online models)"""

import tensorflow as tf
from abc import abstractmethod


def tensor_mul(tensor1, tensor2) -> tf.Tensor:
    min_length = min(len(tensor1.shape.as_list()), len(tensor2.shape.as_list()))
    axes0 = [-i for i in range(1, min_length + 1)]
    axes1 = [i for i in range(min_length)]
    return tf.tensordot(tensor1, tensor2, axes=[axes0, axes1])


class Layer(object):
    """Base layer class"""

    _seed = None
    _activations = {
        'sigmoid': tf.keras.activations.sigmoid,
        'relu': tf.keras.activations.relu,
        'tanh': tf.keras.activations.tanh,
        'linear': tf.keras.activations.linear
    }

    def __init__(self, shape: tuple, name: str = None):
        """

        Args:
            shape (tuple): Output layer shape
            name (str): Name of the layer
        """
        self._shape = shape
        name = self.__class__.__name__ if name is None else name
        self._name = name

    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        with tf.name_scope(self._name):
            return self._build_layer(tensor)

    @abstractmethod
    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        return NotImplemented

    def __repr__(self):
        return "{cls}: {shape}".format(cls=self.__class__.__name__, shape=self._shape)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed: int):
        self._seed = new_seed


def Input(shape: tuple, name: str = 'input'):
    return tf.compat.v1.placeholder(tf.float32, shape, name)


class Dense(Layer):
    """Full connected layer"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', use_bias: bool = True, name: str = None):
        """
        Dense layer initialization.

        Args:
            shape (tuple): Shape of output layer tensor
            activation (str): Activation function of the layer. Initially it is a sigmoid
            use_bias (bool): If True use bias weights
            name (str): Name of the layer
        """
        super().__init__(shape, name)
        self._activation = self._activations[activation]
        self._use_bias = use_bias

    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor_shape = tensor.shape.as_list()

        weights = tf.random.uniform(tuple(reversed(tensor_shape)) + self._shape, -1, 1, seed=self._seed)
        weights = tf.Variable(weights, name='weights')
        result = tensor_mul(tensor, weights)

        if self._use_bias:
            biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed), name='biases')
            result = self._activation(result + biases)
        else:
            result = self._activation(result)
        return result


class Delay(Layer):

    def __init__(self, num_delays: int, name: str = None):
        super().__init__(shape=(None,), name=name)
        assert isinstance(num_delays, int)
        self._num_delays = num_delays

    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor_shape = tuple(tensor.shape.as_list())
        delays = tf.Variable(tf.zeros(shape=(self._num_delays,) + tensor_shape), trainable=False, name='tensor_delays')
        concatenated = tf.concat([[tensor], delays[:-1, ...]], axis=0)
        delays_assigner = tf.compat.v1.assign(delays, concatenated)
        with tf.control_dependencies([delays_assigner]):
            concatenated = concatenated + 0
        return concatenated


class SRNN(Layer):
    """Simple recurrent neural network layer"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', name: str = None):
        super().__init__(shape, name)
        self._state = None
        self._biases = None
        self._input_weights = None
        self._hidden_weights = None
        self._activation = self._activations[activation]

    def _create_variables(self, input_shape):
        self._state = tf.Variable(tf.zeros(self._shape), trainable=False, name='state')

        self._biases = tf.random.uniform(self._shape, -1, 1, seed=self._seed)
        self._biases = tf.Variable(self._biases, name='biases')

        input_shape = tuple(reversed(input_shape)) + self._shape
        self._input_weights = tf.random.uniform(input_shape, -1, 1, seed=self._seed)
        self._input_weights = tf.Variable(self._input_weights, name='input_weights')

        hidden_shape = tuple(reversed(self._shape)) + self._shape
        self._hidden_weights = tf.random.uniform(hidden_shape, -1, 1, seed=self._seed)
        self._hidden_weights = tf.Variable(self._hidden_weights, name='hidden_weights')

    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        self._create_variables(tensor.shape.as_list())
        inputs = tensor_mul(tensor, self._input_weights)
        hidden = tensor_mul(self._state, self._hidden_weights)
        outputs = self._activation(inputs + hidden + self._biases)
        state_assigner = tf.compat.v1.assign(self._state, outputs)
        with tf.control_dependencies([state_assigner]):
            outputs = outputs + 0
        return outputs


class LSTM(Layer):
    """Classical LSTM layer"""

    def __init__(self, shape: tuple, name: str = None):
        super().__init__(shape, name)

    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        state = tf.Variable(tf.zeros(self._shape), trainable=False, name='state')
        hidden = tf.Variable(tf.zeros(self._shape), trainable=False, name='hidden')

        with tf.name_scope('forget_gate'):
            hf = Dense(self._shape, activation='linear', use_bias=False, name='hf')(hidden)
            xf = Dense(self._shape, activation='linear', use_bias=True, name='xf')(tensor)
            forget_gate = self._activations['sigmoid'](hf + xf)

        with tf.name_scope('input_gate'):
            hi = Dense(self._shape, activation='linear', use_bias=False, name='hi')(hidden)
            xi = Dense(self._shape, activation='linear', use_bias=True, name='xi')(tensor)
            input_gate = self._activations['sigmoid'](hi + xi)

        with tf.name_scope('candidate_cell'):
            hc = Dense(self._shape, activation='linear', use_bias=False, name='hc')(hidden)
            xc = Dense(self._shape, activation='linear', use_bias=True, name='xc')(tensor)
            candidate_cell = self._activations['tanh'](hc + xc)

        with tf.name_scope('output_gate'):
            ho = Dense(self._shape, activation='linear', use_bias=False, name='ho')(hidden)
            xo = Dense(self._shape, activation='linear', use_bias=True, name='xo')(tensor)
            output_gate = self._activations['sigmoid'](ho + xo)

        new_state = state * forget_gate + candidate_cell * input_gate
        state_assigner = tf.compat.v1.assign(state, new_state)

        new_hidden = self._activations['tanh'](state) * output_gate
        hidden_assigner = tf.compat.v1.assign(hidden, new_hidden)

        with tf.control_dependencies([state_assigner, hidden_assigner]):
            outputs = new_hidden + 0

        return outputs


class VMLSTM(Layer):
    """Variable Memory LSTM layer"""

    def __init__(self, output_shape: tuple, memory_shape: tuple, name: str = None):
        super().__init__(output_shape, name)
        self._memory_shape = memory_shape

    def _build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        state = tf.Variable(tf.zeros(self._memory_shape), trainable=False, name='state')
        hidden = tf.Variable(tf.zeros(self._shape), trainable=False, name='hidden')

        with tf.name_scope("control_gate"):
            sc = Dense(self._memory_shape, activation='linear', use_bias=False, name='sc')(state)
            hc = Dense(self._memory_shape, activation='linear', use_bias=False, name='sc')(hidden)
            xc = Dense(self._memory_shape, activation='linear', use_bias=True, name='xf')(tensor)
            control_gate = self._activations['sigmoid'](sc + hc + xc)

        with tf.name_scope("recording_gate"):
            hr = Dense(self._memory_shape, activation='linear', use_bias=False, name='sc')(hidden)
            xr = Dense(self._memory_shape, activation='linear', use_bias=True, name='xf')(tensor)
            recording_gate = self._activations['tanh'](hr + xr)

        new_state = state * control_gate + recording_gate * (1 - control_gate)
        state_assigner = tf.compat.v1.assign(state, new_state)

        with tf.name_scope("output_gate"):
            so = Dense(self._shape, activation='linear', use_bias=False, name='sc')(new_state)
            ho = Dense(self._shape, activation='linear', use_bias=False, name='sc')(hidden)
            xo = Dense(self._shape, activation='linear', use_bias=True, name='xf')(tensor)
            output_gate = self._activations['sigmoid'](so + ho + xo)

        hidden_assigner = tf.compat.v1.assign(hidden, output_gate)

        with tf.control_dependencies([state_assigner, hidden_assigner]):
            outputs = output_gate + 0

        return outputs


if __name__ == "__main__":
    pass
