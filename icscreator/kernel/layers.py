"""Neural network layer class"""

import tensorflow as tf
from typing import List


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