import tensorflow as tf
import numpy as np


class System:
    """
    The main class of all models.
    """
    def __init__(self, input_namespace, output_namespace,
                 name='System_0', adjacency_matrix=None, **models):
        """
        Initialization
        :param name: Name of the model
        :param input_namespace: Dictionary of inputs with initial values
        :param output_namespace: Dictionary of outputs
        :param adjacency_matrix: Matrix with connection information of all nodes
        """
        self._name = name
        self._input_namespace = input_namespace
        self._output_namespace = output_namespace
        self._adjacency_matrix = adjacency_matrix
        self._system_model = self._compile_system()

    def _compile_system(self):

        return None

    def __add__(self, other):
        """Add one system to another"""

    def __mul__(self, other): pass

    def __str__(self): pass

    def add_model(self): pass

    def get_model(self): pass

    def set_input_namespace(self, new_input_namespace): self._input_namespace = new_input_namespace

    def set_output_namespace(self, new_output_namespace): self._output_namespace = new_output_namespace

    def set_name(self, new_name): self._name = new_name

    def set_adjacency_matrix(self, adjacency_matrix): self._adjacency_matrix = adjacency_matrix

    def get_input_namespace(self): return self._input_namespace

    def get_output_namespace(self): return self._output_namespace

    def get_name(self): return self._name

    def get_adjacency_matrix(self): return self._adjacency_matrix





class Model(System):
    """
    The system with only one object.
    """



class FNN(Model):
    """
    Feedforward neural network class model.
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
    def __init__(self, structure=(1, 1, 1), input_namespace=None, output_namespace=None, name='FNN'):
        self._set_model_name(name)
        if input_namespace is None:
            input_namespace = ['in{}'.format(i+1) for i in range(structure[0])]
        if output_namespace is None:
            output_namespace = ['out{}'.format(i+1) for i in range(structure[-1])]
        self._set_input_namespace(input_namespace)
        self._set_output_namespace(output_namespace)

        # Network inputs
        x = tf.placeholder(tf.float32, [None, structure[0]], 'input')
        y = tf.placeholder(tf.float32, [None, structure[-1]], 'output')


class RNN(Model):
    """
    Recurrent neural network class model.
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


class LSTM(Model):
    pass


class CNN(Model):
    """
    Convolution neural network class model.
    """
    pass


class BNN(Model):
    """
    Bayesian neural network class model.
    """
    pass


class FuzzyLogic(Model):
    pass


if __name__ == '__main__':
    print("Let's test")
