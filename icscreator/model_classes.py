"""
Old file. It should be rewriten.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Input

from tensorflow.examples.tutorials.mnist import input_data


# Easy way to create structures like in the MatLab
class Structure(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

"""
class Model:

    The main class of all ml_models. Everything is model.

    def __init__(self):

        self._model = None  # Here is one or many ml_models in dictionary

        self._info = Structure()
        self._info.name = None
        self._info.comments = None

        self._structure = Structure()

        self._structure.inputs = Structure()
        self._structure.outputs = Structure()

    def __call__(self, input_data):
        with tf.Session(graph=self._config.graph.graph_model) as sess:
            # load previous session
            saver = tf.train.Saver()
            saver.restore(sess, self._config.save_info.tf)
            data_len = len(input_data)
            result = np.zeros((data_len, self._config.num_outputs))
            for i in range(data_len):
                inp = input_data[i].reshape((1, self._config.num_inputs))
                result[i] = sess.run(self._config.graph.model,
                                     feed_dict={self._config.graph.input: inp})
            return result

    def train(self, input_data, target_data):
        with tf.Session(graph=self._config.graph.graph_model) as sess:
            # Load previous session
            saver = tf.train.Saver()
            saver.restore(sess, self._config.save_info.tf)
            # Parameters initializing
            sess.run(tf.global_variables_initializer())
            data_len = input_data.shape[0]
            loss = 0

            for e in range(self._config.epochs):
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # !!! Add changing learn rate in each step using parabola loss estimating!!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for b in range(data_len // self._config.batch_size):
                    inp = input_data[b*self._config.batch_size:(b+1)*self._config.batch_size]
                    inp = inp.reshape((self._config.batch_size, self._config.num_inputs))

                    tar = target_data[b*self._config.batch_size:(b+1)*self._config.batch_size]
                    tar = tar.reshape((self._config.batch_size, self._config.num_outputs))

                    _, loss = sess.run([self._config.graph.optimizer, self._config.graph.loss],
                                       feed_dict={self._config.graph.input: inp,
                                                  self._config.graph.output: tar})
                print('Epoch = {}, cost = {}'.format(e, loss))

            # Save results
            saver.save(sess, self._config.save_info.tf)

    def save(self):
        with open('{}.pkl'.format(self._config.save_info.path + self._config.name), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path_with_name):
        with open(path_with_name, 'rb') as input:
            model = pickle.load(input)
        return model

    def set_epochs(self, epochs): self._config.epochs = epochs

    def set_learning_rate(self, learn_rate): self._config.learn_rate = learn_rate

    def set_name(self, name):
        self._config.name = name
        self._config.save_info.tf = self._config.save_info.path + name + '/tf/{}'.format(name + '.ckpt')
        self._config.save_info.tb = self._config.save_info.path + name + '/tb/'

    def set_path(self, path):
        self._config.save_info.path = path
        self._config.save_info.tf = path + self._config.name + '/tf/{}'.format(self._config.name + '.ckpt')
        self._config.save_info.tb = path + self._config.name + '/tb/'

    def set_train_algorithm(self, algorithm): self._config.graph.train_alg = algorithm

    def set_loss_function(self, function): self._config.graph.loss_fun = function

    def set_num_inputs(self, num_inputs): self._config.num_inputs = num_inputs

    def set_num_outputs(self, num_outputs): self._config.num_outputs = num_outputs

    def set_batch_size(self, batch_size): self._config.batch_size = batch_size

    def get_epochs(self): return self._config.epochs

    def get_learning_rate(self): return self._config.learn_rate

    def get_name(self): return self._config.name

    def get_path(self): return self._config.save_info.path

    def get_train_algorithm(self): return self._config.graph.train_alg

    def get_loss_function(self): return self._config.graph.loss_fun

    def get_num_inputs(self): return self._config.num_inputs

    def get_num_outputs(self): return self._config.num_outputs

    def get_batch_size(self): return self._config.batch_size

    def get_config(self): return self._config


class System(Model):

    Connections between more than one model are system. Two and more systems can connect in the big system.

    def __init__(self, input_namespace, output_namespace, name='System_0', path='./results/'):

        Initialization
        :param name: Name of the model
        :param input_namespace: Dictionary of inputs with initial values
        :param output_namespace: Dictionary of outputs
        :param adjacency_matrix: Matrix with connection information of all nodes

        super().__init__(name=name, path=path)
        self._config.input_namespace = input_namespace
        self._config.output_namespace = output_namespace

    def add_model(self): pass

    def get_model(self): pass

    def set_input_namespace(self, new_input_namespace): self._config.input_namespace = new_input_namespace

    def set_output_namespace(self, new_output_namespace): self._config.output_namespace = new_output_namespace

    def get_input_namespace(self): return self._config.input_namespace

    def get_output_namespace(self): return self._config.output_namespace


class FNN(Model):

    Feedforward neural network class model.

    def __init__(self, structure=(1, 1, 1),
                 act_funs=(tf.sigmoid, tf.sigmoid),
                 loss_fun=tf.losses.mean_squared_error,
                 train_alg=tf.train.GradientDescentOptimizer,
                 learn_rate=0.001,
                 epochs=1000,
                 name='FNN',
                 path='./results/'):

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
        :param path: Path for saved ml_models and data

        super().__init__(name=name, path=path)
        self._config.num_inputs = structure[0]
        self._config.num_outputs = structure[-1]
        self._config.structure = structure
        self._config.act_funs = act_funs

        self.set_train_algorithm(train_alg)
        self.set_loss_function(loss_fun)
        self.set_learning_rate(learn_rate)
        self.set_epochs(epochs)

        self._config.update(self._graph_creation(self._config))

    def _graph_creation(self, config):

        Helper for creating graph.
        :param config: Neural network configuration information
        :return: Config with added ml_models

        # Saving everything is this graph
        # config.graph.graph_model.as_default()
        with tf.Session(graph=config.graph.graph_model) as sess:
            # Input placeholders
            config.graph.input = tf.placeholder(tf.float32, [None, config.structure[0]], 'input')
            config.graph.output = tf.placeholder(tf.float32, [None, config.structure[-1]], 'output')

            # Creating layers
            model = None
            for i in range(len(config.structure)-1):
                with tf.name_scope('layer_{}'.format(i+1)):
                    weights = tf.Variable(tf.random_uniform([config.structure[i], config.structure[i+1]],
                                                            dtype=tf.float32),
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

            # Save graph session for further using
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().save(sess, config.save_info.tf)

            # Save graph for tensorboard visualization
            tf.summary.FileWriter(config.save_info.tb, graph=config.graph.graph_model)

        return config


class RNN:

    Recurrent neural network class model.

    def __init__(self, structure, connections, names_in=None, names_out=None):

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

    Convolution neural network class model.

    def __init__(self, image_size,
                 depth_lays,
                 full_con_size,
                 window_size,
                 pull_size=2,
                 batch_size=100,
                 learn_rate=0.001,
                 epochs=1000,
                 loss_fun=tf.losses.softmax_cross_entropy,
                 train_alg=tf.train.AdamOptimizer,
                 name='CNN',
                 path='./results/'):


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
            tf.losses.sparse_softmax_cross_entropy
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
        :param path: Path for saved ml_models and data

        super().__init__(name=name, path=path)
        self._config.image_size = image_size
        self._config.depth_lays = [image_size[2]] + depth_lays
        self._config.full_con_size = full_con_size
        self._config.window_size = window_size
        self._config.pull_size = pull_size

        self.set_num_inputs(np.prod(image_size))
        self.set_num_outputs(full_con_size[-1])
        self.set_batch_size(batch_size)
        self.set_learning_rate(learn_rate)
        self.set_epochs(epochs)
        self.set_loss_function(loss_fun)
        self.set_train_algorithm(train_alg)

        self._config.update(self._graph_creation(self._config))

    def _graph_creation(self, config):

        Helper for creating graph.
        :param config: Neural network configuration information
        :return: Config with added ml_models

        # Saving everything in this graph
        # config.graph.graph_model.as_default()
        with tf.Session(graph=config.graph.graph_model) as sess:
            # Input placeholders
            config.graph.input = tf.placeholder(tf.float32, [None, np.prod(config.image_size)], 'input')
            config.graph.output = tf.placeholder(tf.float32, [None, config.full_con_size[-1]], 'output')
            # config.graph.keep_prob = tf.placeholder(tf.float32, 1, 'keep_prob')
            with tf.name_scope(config.name):
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

                # This is only for class prediction. Return position of maximum value
                # model = [tf.argmax(model, axis=1)]

                config.graph.model = model

            # Cost function and train algorithm
            config.graph.loss = config.graph.loss_fun(config.graph.output, config.graph.model)
            config.graph.optimizer = config.graph.train_alg(config.learn_rate).minimize(config.graph.loss)

            # Save graph session for further using
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().save(sess, config.save_info.tf)

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
            if layer_num != len(config.full_con_size) - 1:
                full_con = tf.nn.relu(tf.add(full_con, biases))
            else:
                full_con = tf.nn.softmax(tf.add(full_con, biases))
        return full_con

    def set_image_size(self, image_size):
        self._config.image_size = image_size
        self._config.depth_lays = [image_size[2]] + self._config.depth_lays

    def set_depth_lays(self, depth_lays): self._config.depth_lays = [self._config.image_size[2]] + depth_lays

    def set_full_con_size(self, full_con_size): self._config.full_con_size = full_con_size

    def set_window_size(self, window_size): self._config.window_size = window_size

    def set_pull_size(self, pull_size): self._config.pull_size = pull_size

    def get_image_size(self): return self._config.image_size

    def get_depth_lays(self): return self._config.depth_lays[1:]

    def get_full_con_size(self): return self._config.full_con_size

    def get_window_size(self): return self._config.window_size

    def get_pull_size(self): return self._config.pull_size


class BNN(Model):

    Bayesian neural network class model.
    pass


class FuzzyLogic(Model):
    pass

"""


class Model:
    pass


class CNN(Model):
    def __init__(self,
                 image_shape=(224, 224, 3),
                 parallels=2,
                 depths=(48, 128, 192, 192, 128),
                 windows=(11, 5, 3, 3, 3),
                 strides=(4, 1, 1, 1, 1),
                 pools=(2, 2, 0, 0, 2),
                 intersections=(0, 0, 1, 0, 0),
                 denses=(2048, 2048, 1000)):

        self._structure = Structure()
        self._structure.image_shape = image_shape
        self._structure.parallels = parallels
        self._structure.depths = depths
        self._structure.windows = windows
        self._structure.strides = strides
        self._structure.pools = pools
        self._structure.intersections = intersections

        model = [Input(shape=image_shape)] * parallels
        for layer_num in range(len(depths)):
            depth = depths[layer_num]
            window = windows[layer_num]
            strise = strides[layer_num]
            pool = pools[layer_num]
            intersection = intersections[layer_num]

        model = [layers.Conv2D(filters=10, kernel_size=3, strides=1, padding='same', activation='relu')(x) for x in
                 model]

        self._model = model
        self._model.summary()

    def _construct_model(self):



        return tf.keras.models.Model(input_tensor, model)


if __name__ == '__main__':
    print("Let's test")
    """
    Data_JC = np.genfromtxt('../data/Data_JetCat_P60.csv', delimiter=',')
    steps = 100
    fuel = Data_JC[0:-1:steps, 1] / 4.0
    freq = Data_JC[0:-1:steps, 2] / 200000.0
    temp = Data_JC[0:-1:steps, 3] / 1000.0

    nn = FNN(structure=(1, 5, 1), learn_rate=0.01, epochs=1000)
    nn.train(fuel, freq)
    nn.set_batch_size(10000)
    nn.set_train_algorithm(tf.train.AdamOptimizer)
    # nn.save()
    res = nn(fuel)
    plt.plot(res)
    plt.plot(freq)
    plt.show()

    nn = CNN(image_size=[28, 28, 1], depth_lays=[2, 4], batch_size=1000,
             full_con_size=[100, 10], window_size=[5, 5], pull_size=2)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)

    left = 2.5
    top = 2.5

    fig = plt.figure(figsize=(10, 10))

    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)
        im = np.reshape(mnist.train.images[i, :], [28, 28])

        label = np.argmax(mnist.train.labels[i, :])
        ax.imshow(im, cmap='Greys')
        ax.text(left, top, str(label))
    plt.show()

    nn.train(mnist.train.images, mnist.train.labels)
    print(nn(mnist.train.images[0:4]))
    """
    cnn = CNN(image_shape=(224, 224, 3))
