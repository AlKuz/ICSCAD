import tensorflow as tf
import numpy as np


class NeuralNetwork:
    """
    structure = [2, 3, 1] - network structure with 2 input neurons, 1 hidden
        layer with 3 neurons, and 1 output neuron.
    connections = [in, hid, out] - connection between layers.
        in = [1, 0] - connection between input and hidden layer.
        hid = [[0, 1], [1, 0]] - connection from hidden to output layer, and from
            output to hidden layer as a feedback.
        out = [0, 1] - show what is the layers go outside; here there is only output
            layer go outside. If out = [1, 1], then hidden layer goes outside too.
        If the number of connection more, then 1, then it is delayed connections.
    names = {'in': {'a', 'b'}, 'out': {'c', 'd'}} - names of the inputs and outputs
        variables. Initially this value is None.
    """

    def __init__(self, structure, connections, names_in=None, names_out=None):
        self.structure = structure
        self.connections = connections
        self.graph = tf.Graph()
        self._input = tf.placeholder(tf.float32, [structure(0)], names_in)
        self._output = tf.placeholder(
            tf.float32,
            [sum(np.array(connections[-1]) * np.array(structure[1:]))],
            names_out)
        self._weights = {}
        for i in range(len(structure)-1):
            self._weights[i] = tf.Variable(tf.random_normal([i+1, i]))
