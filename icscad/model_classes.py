import tensorflow as tf
import numpy as np


class MLP:
    """
    Multilayer perceptron
    """
    def __init__(self, structure, act_funs, learn_rate=0.001, epochs=1000):
        """
        :param structure: [2, 3, 1] - network structure with 2 input neurons,
            1 hidden layer with 3 neurons, and 1 output neuron.
        :param act_funs: [tf.sigmoid, tf.tanh] - list of activation functions. Hidden
            layer has sigmoidal activation function, output layer has hyperbolic tangent.
        :param learn_rate: Velocity of training.
        :param epochs: Number of training epochs.

        :type structure: numpy.array
        :type act_funs: [tensorflow_activation_function]
        :type learn_rate: float
        :type epochs: int
        """
        self.structure = structure
        self.act_funs = act_funs

        # Inputs in the tensorflow graph
        input = tf.placeholder("float", [None, self.structure[0]])
        output = tf.placeholder("float", [None, self.structure[-1]])

        # Store layers weight & bias
        self.weights = {}
        self.biases = tf.random_uniform((1, self.structure.shape-1))
        for i in range(len(self.structure)):
            self.weights['w{}'.format(i)] = \
                tf.Variable(tf.random_uniform([self.structure[i+1], self.structure[i]]))



class RNN:
    """
    Recurrent neural network
    """
    def __init__(self, structure, connections, names_in=None, names_out=None):
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
        :param structure:
        :param connections:
        :param names_in:
        :param names_out:
        """
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


if __name__ == '__main__':
    nn = MLP(np.array([1, 5, 1]), [tf.sigmoid, tf.sigmoid])
    print(1)
