import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
Run tensorboard:
for Linux: tensorboard --logdir=path/to/log-directory
for Windows: python -m tensorboard.main
"""


class Structure(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# Not now
class System:
    """
    The main class of all models.
    """
    """
    def __init__(self, input_namespace, output_namespace,
                 name='System_0', adjacency_matrix=None, **models):

        Initialization
        :param name: Name of the model
        :param input_namespace: Dictionary of inputs with initial values
        :param output_namespace: Dictionary of outputs
        :param adjacency_matrix: Matrix with connection information of all nodes

        self._name = name
        self._input_namespace = input_namespace
        self._output_namespace = output_namespace
        self._adjacency_matrix = adjacency_matrix
        self._system_model = self._compile_system()
    """

    def __add__(self, other):
        """Add one system to another. Systems are parallel."""

    def __mul__(self, other):
        """Multiple one system to another. System located each other."""

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


# Not now
class Model(System):
    """
    The system with only one object.
    """


class FNN:
    """
    Feedforward neural network class model.
    """
    def __init__(self, structure=(1, 1, 1),
                 act_funs=(tf.sigmoid, tf.sigmoid),
                 loss_fun=tf.losses.mean_squared_error,
                 train_alg=tf.train.GradientDescentOptimizer,
                 learn_rate=0.001,
                 epochs=1000,
                 name='FNN',
                 path='./results/'):
        """
        :param structure: (2, 3, 1) - network structure with 2 input neurons,
            1 hidden layer with 3 neurons, and 1 output neuron.
        :param act_funs: list of activation functions. Possible activation functions:
            tf.nn.relu
            tf.nn.relu6
            tf.nn.crelu
            tf.nn.elu
            tf.nn.softplus
            tf.nn.softsign
            tf.nn.dropout
            tf.nn.bias_add
            tf.sigmoid
            tf.tanh
        :param loss_fun: Function for calculation faults of neural network. Possible loss functions:
            tf.losses.absolute_difference
            tf.losses.cosine_distance
            tf.losses.huber_loss
            tf.losses.log_loss
            tf.losses.mean_pairwise_squared_error
            tf.losses.mean_squared_error
            -- add other loss functions: loss_function(labels, prediction), where
                labels: The ground truth output tensor, same dimensions as 'predictions'.
                predictions: The predicted outputs.
        :param train_alg: Algorithm for minimization network fault. Possible algorithms:
            tf.train.GradientDescentOptimizer
            tf.train.AdadeltaOptimizer
            tf.train.AdagradOptimizer
            tf.train.AdagradDAOptimizer
            tf.train.MomentumOptimizer
            tf.train.AdamOptimizer
            tf.train.FtrlOptimizer
            tf.train.ProximalGradientDescentOptimizer
            tf.train.ProximalAdagradOptimizer
            tf.train.RMSPropOptimizer
        :param learn_rate: Velocity of training.
        :param epochs: Number of training epochs.
        :param name: Name of the model. It is using for creating folder with saved model.
        :param path: Path for saved models and data
        """
        self._config = Structure()
        self._config.graph = Structure()
        self._config.save_info = Structure()

        self._config.structure = structure
        self._config.act_funs = act_funs
        self._config.learn_rate = learn_rate
        self._config.epochs = epochs
        self._config.name = name

        self._config.graph.graph_model = tf.Graph()
        self._config.graph.input = None
        self._config.graph.output = None
        self._config.graph.model = None
        self._config.graph.optimizer = None
        self._config.graph.loss = 0
        self._config.graph.train_alg = train_alg
        self._config.graph.loss_fun = loss_fun

        # self._config.save_info.saver = tf.train.Saver()
        self._config.save_info.path = path
        self._config.save_info.tf = path + name + '/tf/{}'.format(name + '.ckpt')
        self._config.save_info.tb = path + name + '/tb/'
        self._config.save_info.model = None  # Save this object

        self._config.update(self._graph_creation(self._config))

    def _graph_creation(self, config):
        """
        Helper for creating graph.
        :param config: Neural network configuration information
        :return: Config with added models
        """
        # Saving everything is this graph
        config.graph.graph_model.as_default()
        # Input placeholders
        config.graph.input = tf.placeholder(tf.float32, [None, config.structure[0]], 'input')
        config.graph.output = tf.placeholder(tf.float32, [None, config.structure[-1]], 'output')

        # Creating layers
        model = None
        for i in range(len(config.structure)-1):
            with tf.name_scope('layer_{}'.format(i+1)):
                weights = tf.Variable(tf.random_uniform([config.structure[i], config.structure[i+1]], dtype=tf.float32),
                                      name='weights_{}'.format(i+1))
                biases = tf.Variable(tf.random_uniform([config.structure[i+1]], dtype=tf.float32),
                                     name='biases_{}'.format(i+1))
                # For first layer input is input placeholder
                if i == 0: model = config.graph.input
                model = config.act_funs[i](tf.add(tf.matmul(model, weights), biases))
        config.graph.model = model

        # Cost function
        config.graph.loss = config.graph.loss_fun(config.graph.output, config.graph.model)
        config.graph.optimizer = config.graph.train_alg(config.learn_rate).minimize(config.graph.loss)

        # Save graph for tensorboard visualization
        with tf.Session(graph=config.graph.graph_model):
            tf.summary.FileWriter(config.save_info.tb, graph=config.graph.graph_model)

        return config

    def train(self, input_data, target_data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            data_len = len(input_data)
            input_data = input_data.reshape((data_len, self._config.structure[0]))
            target_data = target_data.reshape((data_len, self._config.structure[-1]))
            for e in range(self._config.epochs):
                _, loss = sess.run([self._config.graph.optimizer, self._config.graph.loss],
                                   feed_dict={self._config.graph.input: input_data,
                                              self._config.graph.output: target_data})
                print('Epoch = {}, cost = {}'.format(e, loss))
            # Save file
            saver = tf.train.Saver()
            saver.save(sess, self._config.save_info.tf)

    def __call__(self, input_data):
        with tf.Session() as sess:
            # load previous session
            saver = tf.train.Saver()
            saver.restore(sess, self._config.save_info.tf)
            data_len = len(input_data)
            result = np.zeros((data_len, self._config.structure[-1]))
            input_data = input_data.reshape((data_len, self._config.structure[0]))
            for i in range(data_len):
                inputs = input_data[i, :].reshape((1, self._config.structure[0]))
                result[i] = sess.run(self._config.graph.model,
                                     feed_dict={self._config.graph.input: inputs})
            return result

    def get_config(self): return self._config


class RNN:
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
    def __init__(self, image_size,
                 depth_lays,
                 full_con_size,
                 window_size,
                 pull_size=2,
                 batch_size=128,
                 learn_rate=0.001,
                 epochs=1000,
                 loss_fun=tf.losses.mean_squared_error,
                 train_alg=tf.train.GradientDescentOptimizer,
                 name='CNN',
                 path='./results/'):
        """

        :param image_size: [1024, 768, 3] - width, height, depth
        :param depth_lays: [6, 10] - depth in each conv layer
        :param full_con_size: [100, 10] - number of neurons in the hidden layers, number of output classes
        :param window_size: [5, 5]
        :param pull_size: 2 - pooling in each layer
        :param batch_size: Number of images in the one step during training algorithm
        :param learn_rate: Velocity of training
        :param epochs: Number of training epochs
        :param loss_fun: Function for calculation faults of neural network. Possible loss functions:
            tf.losses.absolute_difference
            tf.losses.cosine_distance
            tf.losses.huber_loss
            tf.losses.log_loss
            tf.losses.mean_pairwise_squared_error
            tf.losses.mean_squared_error
            -- add other loss functions: loss_function(labels, prediction), where
                labels: The ground truth output tensor, same dimensions as 'predictions'.
                predictions: The predicted outputs.
        :param train_alg: Algorithm for minimization network fault. Possible algorithms:
            tf.train.GradientDescentOptimizer
            tf.train.AdadeltaOptimizer
            tf.train.AdagradOptimizer
            tf.train.AdagradDAOptimizer
            tf.train.MomentumOptimizer
            tf.train.AdamOptimizer
            tf.train.FtrlOptimizer
            tf.train.ProximalGradientDescentOptimizer
            tf.train.ProximalAdagradOptimizer
            tf.train.RMSPropOptimizer
        :param name: Name of the model. It is using for creating folder with saved model.
        :param path: Path for saved models and data
        """
        self._config = Structure()
        self._config.graph = Structure()
        self._config.save_info = Structure()

        self._config.image_size = image_size
        self._config.depth_lays = [image_size[2]] + depth_lays
        self._config.full_con_size = full_con_size
        self._config.window_size = window_size
        self._config.pull_size = pull_size
        self._config.batch_size = batch_size
        self._config.learn_rate = learn_rate
        self._config.epochs = epochs
        self._config.name = name

        self._config.graph.graph_model = tf.Graph()
        self._config.graph.input = None
        self._config.graph.output = None
        self._config.graph.keep_prob = None
        self._config.graph.model = None
        self._config.graph.optimizer = None
        self._config.graph.loss = 0
        self._config.graph.train_alg = train_alg
        self._config.graph.loss_fun = loss_fun

        self._config.save_info.path = path
        self._config.save_info.tf = path + name + '/tf/{}'.format(name + '.ckpt')
        self._config.save_info.tb = path + name + '/tb/'
        self._config.save_info.model = None  # Save this object

        self._config.update(self._graph_creation(self._config))

    def _graph_creation(self, config):
        """
        Helper for creating graph.
        :param config: Neural network configuration information
        :return: Config with added models
        """
        # Saving everything in this graph
        # config.graph.graph_model.as_default()
        with tf.Session(graph=config.graph.graph_model):
            # Input placeholders
            config.graph.input = tf.placeholder(tf.float32, [None, np.prod(config.image_size)], 'input')
            config.graph.output = tf.placeholder(tf.float32, [None, config.full_con_size[-1]], 'output')
            config.graph.keep_prob = tf.placeholder(tf.float32, 1, 'keep_prob')

            # Creating layers
            model = tf.reshape(config.graph.input, shape=[-1]+config.image_size, name='reshaping')

            # Creating convolution layers
            for i in range(1, len(config.depth_lays)):
                model = self._conv(model, i, config)

            # Creating fully connection layers
            config.full_con_size = [np.prod(model.get_shape().as_list()[1:])] + config.full_con_size
            model = tf.reshape(model, shape=[-1, config.full_con_size[0]], name='reshaping')
            for i in range(1, len(config.full_con_size)):
                model = self._full_con(model, i, config)

            config.graph.model = model

            # Cost function
            config.graph.loss = config.graph.loss_fun(config.graph.output, config.graph.model)
            config.graph.optimizer = config.graph.train_alg(config.learn_rate).minimize(config.graph.loss)

            # Save graph for tensorboard visualization

            tf.summary.FileWriter(config.save_info.tb, graph=config.graph.graph_model)

        return config

    def _conv(self, inp, layer_num, config, strides=1):
        with tf.name_scope('Conv_{}'.format(layer_num)):
            shape = config.window_size + config.depth_lays[layer_num-1:layer_num+1]
            weights = tf.Variable(tf.random_uniform(shape, -1, 1, tf.float32), name='weights')
            biases = tf.Variable(tf.random_uniform([config.depth_lays[layer_num]], -1, 1, tf.float32), name='biases')
            conv = tf.nn.conv2d(inp, weights, strides=[1, strides, strides, 1], padding='SAME')
            conv = tf.nn.relu(tf.add(conv, biases))
            k = config.pull_size
            conv = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            return conv

    def _full_con(self, inp, layer_num, config):
        with tf.name_scope('Full_con_{}'.format(layer_num)):
            shape = config.full_con_size[layer_num-1:layer_num+1]
            weights = tf.Variable(tf.random_uniform(shape, -1, 1, tf.float32), name='weights')
            biases = tf.Variable(tf.random_uniform([config.full_con_size[layer_num]], -1, 1, tf.float32), name='biases')
            full_con = tf.matmul(inp, weights)
            full_con = tf.nn.relu(tf.add(full_con, biases))
            return full_con


    def __call__(self, *args, **kwargs):
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
    # tf.reduce_mean(-1 * (self.net_out * tf.log(self.model) + (1 - self.net_out) * tf.log(1 - self.model)))
    Data_JC = np.genfromtxt('../data/Data_JetCat_P60.csv', delimiter=',')
    steps = 1000
    fuel = Data_JC[0:-1:steps, 1] / 4.0
    freq = Data_JC[0:-1:steps, 2] / 200000.0
    temp = Data_JC[0:-1:steps, 3] / 1000.0

    # nn = FNN()
    # nn.train(fuel, freq)
    # res = nn(fuel)
    # plt.plot(res)
    # plt.plot(freq)
    # plt.show()

    nn = CNN(image_size=[1024, 768, 3], depth_lays=[1, 1, 1],
             full_con_size=[1000, 10], window_size=[5, 5], pull_size=4)
    print()
