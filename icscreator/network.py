"""
Old file
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Run tensorboard:
for Linux: tensorboard --logdir=path/to/log-directory
for Windows: python -m tensorboard.main
"""

"""
class RNN:
    def __init__(self, structure, delays):
        self.data = dict()
        self.data['inputs'] = structure[0]
        self.data['outputs'] = structure[-1]
        self.data['delays'] = delays
        for s in range(len(structure[1:-1])):
            self.data['hidden{}'.format(s)] = structure[s+1]
        self.graph = tf.Graph()
        with tf.Session(graph=self.graph) as sess:
            # Inputs in the graph
            inputs = tf.placeholder(dtype=tf.float32, shape=[1, structure[0]], name='inputs')
            outputs = tf.placeholder(dtype=tf.float32, shape=[1, structure[-1]], name='outputs')
            outdelays = tf.placeholder(dtype=tf.float32, shape=[delays, structure[-1]], name='outdelays')
            layer_result = tf.add(tf.matmul(tf.Variable(tf.random_uniform((structure[1], structure[0])), dtype=tf.float32, name='weights0'), inputs),
                                  tf.reduce_sum(tf.matmul(tf.Variable(tf.random_uniform((structure[1], structure[-1])), dtype=tf.float32, name='weightsd'), outdelays), axis=0))
            layer_result = tf.sigmoid(layer_result)
            for layer in range(len(structure)-2):
                layer_result = tf.sigmoid(tf.matmul(tf.Variable(tf.random_uniform((structure[layer+2], structure[layer+1])), dtype=tf.float32, name='weights{}'.format(layer+1)), layer_result))
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./logs/rnn")
            writer.add_graph(self.graph)
"""

"""
class RNN:
    def __init__(self, structure, delays):
        self.data = dict()
        self.data['inputs'] = structure[0]
        self.data['outputs'] = structure[-1]
        self.data['delays'] = delays
        for s in range(len(structure[1:-1])):
            self.data['hidden{}'.format(s)] = structure[s]
        self.graph = tf.Graph()
        self.layer_result = None

        # Building network graph
        with tf.Session(graph=self.graph) as sess:
            # Inputs in the graph
            inputs = tf.placeholder(dtype=tf.float32, shape=[structure[0], 1], name='inputs')
            self.inputs = inputs
            outputs = tf.placeholder(dtype=tf.float32, shape=[structure[-1], 1], name='outputs')
            self.outputs = outputs

            # Context neurons
            with tf.name_scope('context'):
                context = []
                for d in range(delays):
                    context.append(tf.Variable(tf.zeros((structure[-1], 1)), dtype=tf.float32, name='context_weights_{}'.format(d)))

            # First hidden layer
            with tf.name_scope('hidden1'):
                layer_result = tf.sigmoid(tf.matmul(tf.Variable(tf.random_uniform((structure[1], structure[0] + structure[-1] * delays)), dtype=tf.float32, name='weights0'), tf.concat([inputs, tf.concat(context, 0)], 0)))

            # Output layer
            with tf.name_scope('output'):
                layer_result = tf.sigmoid(tf.matmul(tf.Variable(tf.random_uniform((structure[-1], structure[1])), dtype=tf.float32, name='weights1'), layer_result))

            # Delays
            # Test it!!!
            for d in range(len(context)):
                if d == len(context)-1:
                    context[d].assign(layer_result)
                else:
                    context[d].assign(context[d+1])

            sess.run(tf.global_variables_initializer())
            self.layer_result = layer_result
            self.loss = tf.reduce_mean(tf.square(layer_result - self.outputs))
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
            writer = tf.summary.FileWriter("./logs/rnn")
            writer.add_graph(self.graph)

    def __call__(self, data):
        result = np.zeros((len(data), 1))
        with tf.Session(graph=self.graph) as sess:
            for i in range(len(data)):
                sess.run(tf.global_variables_initializer())
                result[i] = sess.run(self.layer_result, feed_dict={self.inputs: [[data[i]]]})
        return result

    def train(self, input, target):
        plt.ion()
        loss = np.zeros((len(input), 1))
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                for r in range(len(input)):
                    feed_dict = {self.inputs: [[input[r]]], self.outputs: [[target[r]]]}
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                plt.plot(i, loss, 'k.')
                plt.pause(0.00000001)
        plt.ioff()

    def show(self):
        pass
"""

"""
class NeuralNetwork:
    def __init__(self, structure=(1, 1, 1), connections=((0, 1), (1, 0)), epochs=1000, batch_size=100,
                 training_algorithm=None, learning_rate=0.01, min_fault=0.001):


        :param structure: (1, 3, 1)
        :param connections:
        :param epochs:
        :param batch_size:
        :param training_algorithm:
        
        self.__structure = structure
        self.__connections = connections
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__training_algorithm = training_algorithm
        self.__learning_rate = learning_rate
        self.__min_fault = min_fault
        self.__graph = tf.Graph()
        self.__network_result = None

        with self.__graph.as_default():
            # Create input and output placeholders to store values while training
            inputs = tf.placeholder(shape=[None, structure[0]], dtype=tf.float32, name='inputs')
            self.__inputs = inputs
            outputs = tf.placeholder(shape=[None, structure[-1]], dtype=tf.float32, name='outputs')
            self.__outputs = outputs
            layer_result = None

            for i in range(len(structure)-1):
                with tf.name_scope('layer_{}'.format(i+1)):
                    weights = tf.Variable(tf.random_uniform([structure[i], structure[i+1]], dtype=tf.float32),
                                          name='weights_{}'.format(i+1))
                    biases = tf.Variable(tf.random_uniform(shape=[structure[i+1]], dtype=tf.float32),
                                         name='biases_{}'.format(i+1))
                    if i == 0:
                        layer_input = inputs
                    else:
                        layer_input = layer_result
                    layer_result = tf.sigmoid(tf.add(tf.matmul(layer_input, weights), biases),
                                              name='layer_{}_result'.format(i + 1))
            self.__network_result = layer_result
            self.__loss = tf.reduce_mean(-1 * (self.__outputs * tf.log(self.__network_result) + (1 - self.__outputs) * tf.log(1 - self.__network_result)))
            # self.__loss = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.__network_result, self.__outputs), 2)))

        with tf.Session(graph=self.__graph):
            tf.summary.FileWriter("./logs/rnn", graph=self.__graph)

    def __call__(self, input_data):
        data_len = len(input_data)
        result = np.zeros((data_len, self.__structure[-1]))
        input_data = input_data.reshape((data_len, self.__structure[0]))
        with tf.Session(graph=self.__graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(data_len):
                inputs = input_data[i, :].reshape((1, self.__structure[0]))
                result[i] = sess.run(self.__network_result, feed_dict={self.__inputs: inputs})
        return result

    def train(self, input_data, target_data):
        data_len = len(input_data)
        input_data = input_data.reshape((data_len, self.__structure[0]))
        target_data = target_data.reshape((data_len, self.__structure[-1]))
        with tf.Session(graph=self.__graph) as sess:
            sess.run(tf.global_variables_initializer())
            optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(self.__loss)
            for e in range(self.__epochs):
                loss = 0.0
                for i in range(data_len // self.__batch_size):
                    feed_dict = {
                        self.__inputs: input_data[i*self.__batch_size:(i+1)*self.__batch_size, :],
                        self.__outputs: target_data[i*self.__batch_size:(i+1)*self.__batch_size, :]
                    }
                    _, loss = sess.run([optimizer, self.__loss], feed_dict=feed_dict)
                    loss += loss
                if e % 10 == 0:
                    print('Epochs: {}, loss: {}'.format(e, loss))
                if loss <= self.__min_fault: break

    def set_batch_size(self, new_batch_size):
        pass

    def set_training_steps(self, new_training_steps):
        pass

    def get_info(self):
        pass

    def get_graph(self):
        pass
"""


class RNN:
    def __init__(self, structure, connections, epochs=1000, learning_rate=0.001):
        self.structure = structure
        self.connections = connections
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Network inputs
        self.net_in = tf.placeholder(tf.float32, [None, structure[0]], 'input')
        self.net_out = tf.placeholder(tf.float32, [None, structure[-1]], 'output')
        # Network model creation
        # First layer
        with tf.name_scope('layer_1'):
            weights = tf.Variable(tf.random_uniform([structure[0], structure[1]], dtype=tf.float32), name='weights_1')
            biases = tf.Variable(tf.random_uniform([structure[1]], dtype=tf.float32), name='biases_1')
            self.model = tf.sigmoid(tf.add(tf.matmul(self.net_in, weights), biases), name='res_1')
        # Other layers
        for i in range(2, len(structure)):
            with tf.name_scope('layer_{}'.format(i)):
                weights = tf.Variable(tf.random_uniform([structure[i-1], structure[i]], dtype=tf.float32),
                                      name='weights_{}'.format(i))
                biases = tf.Variable(tf.random_uniform(shape=[structure[i]], dtype=tf.float32),
                                     name='biases_{}'.format(i))
                self.model = tf.sigmoid(tf.add(tf.matmul(self.model, weights), biases), name='res_{}'.format(i))
        # Cost function (cross entropy)
        self.cost_fun = tf.reduce_mean(-1 * (self.net_out * tf.log(self.model) + (1 - self.net_out) * tf.log(1 - self.model)))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_fun)
        # The file path to save the data
        self.save_file = './ml_models/rnn.ckpt'
        # Class used to do the saving
        self.saver = tf.train.Saver()

    def __call__(self, input_data):
        with tf.Session() as sess:
            # load previous session
            self.saver.restore(sess, self.save_file)
            data_len = len(input_data)
            result = np.zeros((data_len, self.structure[-1]))
            input_data = input_data.reshape((data_len, self.structure[0]))
            for i in range(data_len):
                inputs = input_data[i, :].reshape((1, self.structure[0]))
                result[i] = sess.run(self.model, feed_dict={self.net_in: inputs})
            return result

    def train(self, input_data, target_data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            data_len = len(input_data)
            input_data = input_data.reshape((data_len, self.structure[0]))
            target_data = target_data.reshape((data_len, self.structure[-1]))
            for e in range(self.epochs):
                _, cost = sess.run([self.optimizer, self.cost_fun], feed_dict={self.net_in: input_data,
                                                                               self.net_out: target_data})
                print('Epoch = {}, cost = {}'.format(e, cost))
            # Save file
            self.saver.save(sess, self.save_file)


if __name__ == "__main__":

    Data_JC = np.genfromtxt('Data_JC.csv', delimiter=',')
    steps = 1000
    fuel = Data_JC[0:-1:steps, 1] / 4.0
    freq = Data_JC[0:-1:steps, 2] / 200000.0
    temp = Data_JC[0:-1:steps, 3]
    # rnn = RNN(structure=(1, 15, 1), delays=3)
    # rnn.train(input=fuel, target=freq)
    # freq_nn = rnn(fuel)
    # plt.plot(freq_nn)
    # plt.show()

    input_XOR = np.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]])

    target_XOR = np.array([[0], [1], [1], [0]])

    nn = RNN(structure=(1, 2, 1), connections=None, learning_rate=0.1, epochs=20000)
    nn.train(fuel, freq)
    res = nn(fuel)
    plt.plot(res)
    plt.plot(freq)
    plt.show()
