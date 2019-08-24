"""Neural network layer classes (only for online models)"""

import tensorflow as tf
from typing import List
from abc import abstractmethod


def name_scope(method):
    def wrapper(self, *args):
        with tf.name_scope(self._name):
            return method(self, *args)
    return wrapper


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
        'tanh': tf.keras.activations.tanh,
        'relu': tf.keras.activations.relu,
        'linear': lambda x: x
    }

    def __init__(self, shape: tuple, name: str):
        """

        Args:
            shape (tuple): Output layer shape
            name (str): Name of the layer
            activation (str): Activation function of the layer. Initially it is a sigmoid
        """
        self._shape = shape
        self._name = name

    def __call__(self, *args) -> tf.Tensor:
        return tf.keras.layers.Lambda(self.build_layer, output_shape=self._shape, name=self._name)(args[0])

    @abstractmethod
    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
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
        super().__init__(shape, name)

    def __call__(self) -> tf.Tensor:
        return tf.keras.layers.Input(tensor=tf.compat.v1.placeholder(tf.float32, self._shape, self._name),
                                     name=self._name)


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
        super().__init__(shape, name)
        self._activation = self._activations[activation]
        self._use_bias = use_bias

    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor_shape = tensor.shape.as_list()

        weights = tf.random.uniform(tuple(reversed(tensor_shape)) + self._shape, -1, 1, seed=self._seed)
        weights = tf.Variable(weights, name='weights')
        result = tensor_mul(tensor, weights)

        if self._use_bias:
            biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed), name='biases')
            result = self._activation(result + biases)

        result = self._activation(result)
        return result


class Delay(Layer):

    def __init__(self, num_delays: int, name='delay'):
        super().__init__(shape=(None,), name=name)
        assert isinstance(num_delays, int)
        self._num_delays = num_delays

    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor_shape = tuple(tensor.shape.as_list())
        delays = tf.Variable(tf.zeros(shape=(self._num_delays,) + tensor_shape), trainable=False, name='tensor_delays')
        concatenated = tf.concat([[tensor], delays[:-1, ...]], axis=0)
        delays = tf.compat.v1.assign(delays, concatenated)
        return delays


class SRNN(Layer):
    """Simple recurrent neural network layer"""

    def __init__(self, shape: tuple, activation: str = 'sigmoid', name: str = 'srnn'):
        super().__init__(shape, name)
        self._activation = self._activations[activation]

    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        inlet_shape = tensor.shape.as_list()
        state = tf.Variable(tf.zeros(self._shape), trainable=False, name='state')

        biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed), name='biases')
        inlet_weights = tf.random.uniform(tuple(reversed(inlet_shape)) + self._shape, -1, 1, seed=self._seed)
        inlet_weights = tf.Variable(inlet_weights, name='inlet_weights')
        hidden_weights = tf.random.uniform(tuple(reversed(self._shape)) + self._shape, -1, 1, seed=self._seed)
        hidden_weights = tf.Variable(hidden_weights, name='hidden_weights')

        inlet = tensor_mul(tensor, inlet_weights)
        hidden = tensor_mul(state, hidden_weights)
        outlet = self._activation(inlet + hidden + biases)
        state = tf.compat.v1.assign(state, outlet)
        return state


class LSTM(Layer):
    """Classical LSTM layer"""

    def __init__(self, shape: tuple, name: str = 'lstm'):
        super().__init__(shape, name)

    @name_scope
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        state = tf.Variable(tf.zeros(self._shape), trainable=False, name='state')
        hidden = tf.Variable(tf.zeros(self._shape), trainable=False, name='hidden')

        with tf.name_scope('forget_gate'):
            hf = Dense(self._shape, activation='linear', use_bias=False, name='hidden')(hidden)
            xf = Dense(self._shape, activation='linear', use_bias=True, name='input')(tensor)
            forget_gate = self._activations['sigmoid'](hf + xf)

        with tf.name_scope('input_gate'):
            hi = Dense(self._shape, activation='linear', use_bias=False, name='hidden')(hidden)
            xi = Dense(self._shape, activation='linear', use_bias=True, name='input')(tensor)
            input_gate = self._activations['sigmoid'](hi + xi)

        with tf.name_scope('candidate_cell'):
            hc = Dense(self._shape, activation='linear', use_bias=False, name='hidden')(hidden)
            xc = Dense(self._shape, activation='linear', use_bias=True, name='input')(tensor)
            candidate_cell = self._activations['tanh'](hc + xc)

        with tf.name_scope('output_gate'):
            ho = Dense(self._shape, activation='linear', use_bias=False, name='hidden')(hidden)
            xo = Dense(self._shape, activation='linear', use_bias=True, name='input')(tensor)
            output_gate = self._activations['sigmoid'](ho + xo)

        new_state = state * forget_gate + candidate_cell * input_gate
        state = tf.assign(state, new_state)

        new_hidden = self._activations['tanh'](state) * output_gate
        hidden = tf.assign(hidden, new_hidden)

        return hidden


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
    inp = Input(shape=(3, 4, 5))()
    model1 = Dense(shape=(6, 7, 8))(inp)
    model1 = tf.keras.models.Model(inp, model1)
    model1.summary()
    model1.save('./model1.hdf5')
    model2 = SRNN(shape=(6, 7, 8))(inp)
    model2 = tf.keras.models.Model(inp, model2)
    model2.summary()
    model3 = Delay(num_delays=5)(inp)
    model3 = tf.keras.models.Model(inp, model3)
    model3.summary()
    # model3 = LSTM(shape=(6, 7, 8))(inp)
