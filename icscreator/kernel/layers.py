"""Neural network layer classes (only for online models)"""

import tensorflow as tf
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

    def __init__(self, input_shape: tuple, output_shape: tuple, name: str):
        """

        Args:
            shape (tuple): Output layer shape
            name (str): Name of the layer
            activation (str): Activation function of the layer. Initially it is a sigmoid
        """
        assert len(input_shape) > 1, "Input shape contains batch shape in the first dimension"
        self._input_shape = input_shape[1:]
        self._output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape
        self._name = name

    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        with tf.keras.backend.get_session().as_default():
            result = tf.keras.layers.Lambda(lambda t: tf.compat.v2.map_fn(self.build_layer, t),
                                      name=self._name)(tensor)
            tf.initialize_all_variables().run()
        return result

    @abstractmethod
    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        raise Exception("Method doesn't implemented")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed: int):
        self._seed = new_seed


def Input(shape: tuple, name: str = 'input'):
    return tf.keras.layers.Input(tensor=tf.compat.v1.placeholder(tf.float32, shape, name), name=name)


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
        weights = tf.Variable(weights, name=self._name + '_' + 'weights')
        result = tensor_mul(tensor, weights)

        if self._use_bias:
            biases = tf.Variable(tf.random.uniform(self._shape, -1, 1, seed=self._seed),
                                 name=self._name + '_' + 'biases')
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
        delays = tf.Variable(tf.zeros(shape=(self._num_delays,) + tensor_shape), trainable=False,
                             name=self._name + '_' + 'tensor_delays')
        concatenated = tf.concat([[tensor], delays[:-1, ...]], axis=0)
        delays = tf.compat.v1.assign(delays, concatenated)
        return delays


class SRNN(Layer):
    """Simple recurrent neural network layer"""

    def __init__(self, input_shape: tuple, output_shape: tuple, activation: str = 'sigmoid', name: str = 'srnn'):
        super().__init__(input_shape, output_shape, name)
        self._state = None
        self._biases = None
        self._inlet_weights = None
        self._hidden_weights = None
        self._activation = self._activations[activation]
        self._create_variables()

    def _create_variables(self):
        self._state = tf.Variable(tf.zeros(self._output_shape), trainable=False, name=self._name + '_' + 'state')

        self._biases = tf.random.uniform(self._output_shape, -1, 1, seed=self._seed)
        self._biases = tf.Variable(self._biases, name=self._name + '_' + 'biases')

        inlet_shape = tuple(reversed(self._input_shape)) + self._output_shape
        self._inlet_weights = tf.random.uniform(inlet_shape, -1, 1, seed=self._seed)
        self._inlet_weights = tf.Variable(self._inlet_weights, name=self._name + '_' + 'inlet_weights')

        hidden_shape = tuple(reversed(self._output_shape)) + self._output_shape
        self._hidden_weights = tf.random.uniform(hidden_shape, -1, 1, seed=self._seed)
        self._hidden_weights = tf.Variable(self._hidden_weights, name=self._name + '_' + 'hidden_weights')

    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        inlet = tensor_mul(tensor, self._inlet_weights)
        hidden = tensor_mul(self._state, self._hidden_weights)
        outlet = self._activation(inlet + hidden + self._biases)
        self._state = tf.compat.v1.assign(self._state, outlet)
        return self._state


class LSTM(Layer):
    """Classical LSTM layer"""

    def __init__(self, shape: tuple, name: str = 'lstm'):
        super().__init__(shape, name)

    def build_layer(self, tensor: tf.Tensor) -> tf.Tensor:
        state = tf.Variable(tf.zeros(self._output_shape), trainable=False, name=self._name + '_' + 'state')
        hidden = tf.Variable(tf.zeros(self._output_shape), trainable=False, name=self._name + '_' + 'hidden')

        with tf.name_scope('forget_gate'):
            hf = Dense(self._output_shape, activation='linear', use_bias=False,
                       name=self._name + '_' + 'hf').build_layer(hidden)
            xf = Dense(self._output_shape, activation='linear', use_bias=True,
                       name=self._name + '_' + 'xf').build_layer(tensor)
            forget_gate = self._activations['sigmoid'](hf + xf)

        with tf.name_scope('input_gate'):
            hi = Dense(self._output_shape, activation='linear', use_bias=False,
                       name=self._name + '_' + 'hi').build_layer(hidden)
            xi = Dense(self._output_shape, activation='linear', use_bias=True,
                       name=self._name + '_' + 'xi').build_layer(tensor)
            input_gate = self._activations['sigmoid'](hi + xi)

        with tf.name_scope('candidate_cell'):
            hc = Dense(self._output_shape, activation='linear', use_bias=False,
                       name=self._name + '_' + 'hc').build_layer(hidden)
            xc = Dense(self._output_shape, activation='linear', use_bias=True,
                       name=self._name + '_' + 'xc').build_layer(tensor)
            candidate_cell = self._activations['tanh'](hc + xc)

        with tf.name_scope('output_gate'):
            ho = Dense(self._output_shape, activation='linear', use_bias=False,
                       name=self._name + '_' + 'ho').build_layer(hidden)
            xo = Dense(self._output_shape, activation='linear', use_bias=True,
                       name=self._name + '_' + 'xo').build_layer(tensor)
            output_gate = self._activations['sigmoid'](ho + xo)

        new_state = state * forget_gate + candidate_cell * input_gate
        state = tf.compat.v1.assign(state, new_state)

        new_hidden = self._activations['tanh'](state) * output_gate
        hidden = tf.compat.v1.assign(hidden, new_hidden)

        return hidden


if __name__ == "__main__":
    inp = Input(shape=(3, 4, 5))

    model1 = Dense(shape=(6, 7, 8))(inp)
    model1 = tf.keras.models.Model(inp, model1)
    model1.summary()

    model2 = SRNN(output_shape=(6, 7, 8))(inp)
    model2 = tf.keras.models.Model(inp, model2)
    model2.summary()

    model3 = Delay(num_delays=5)(inp)
    model3 = tf.keras.models.Model(inp, model3)
    model3.summary()

    model4 = LSTM(shape=(6, 7, 8))(inp)
    model4 = tf.keras.models.Model(inp, model4)
    model4.summary()
