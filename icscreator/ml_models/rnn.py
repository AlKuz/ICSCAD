"""
Recurrent neural networks:

ElmanNetwork
"""
import tensorflow as tf


class NeuralNetwork(object):
    """Base class of neural networks"""

    def __init__(self, seed=None):
        self._seed = seed

    def _layer(self, tensor: tf.Tensor, num_outputs, activation=tf.sigmoid, name='layer'):
        num_inputs = int(tensor.shape[1])
        with tf.name_scope(name):
            biases = tf.Variable(tf.random.uniform([num_outputs], -1, 1, seed=self._seed, name='biases'))
            weights = tf.Variable(tf.random.uniform([num_inputs, num_outputs], -1, 1, seed=self._seed, name='weights'))
            result = activation(tf.matmul(tensor, weights) + biases)
        return result


class ElmanNetwork(NeuralNetwork):
    """Elman network"""

    def __init__(self, inputs: int, hiddens: int, outputs: int, seed=None):
        super().__init__(seed)
        self._inputs = inputs
        self._hiddens = hiddens
        self._outputs = outputs
        self._graph = tf.Graph()
        self._model_inputs = None
        self._model_outputs = None
        self._create_model()

    def _create_model(self):
        with self._graph.as_default():
            self._model_inputs = tf.compat.v1.placeholder(tf.float32, [1, self._inputs], 'inputs')
            state = tf.Variable(tf.zeros([1, self._hiddens]), trainable=False, name='state')

            hiddens = self._layer(tf.concat([self._model_inputs, state], axis=1), self._hiddens, name='hidden')
            hiddens = tf.assign(state, hiddens)
            self._model_outputs = self._layer(hiddens, self._outputs, name='output')

            writer = tf.summary.FileWriter('tb', tf.get_default_graph())

    def predict(self, data):
        result = []
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.initialize_all_variables())
            for i in data:
                r = sess.run(self._model_outputs, feed_dict={self._model_inputs: [i]})
                result.append(list(r))
        return result

    @property
    def hiddens(self):
        return self._hiddens

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


if __name__ == "__main__":
    data = [[1, 2, 3]] * 10
    model = ElmanNetwork(3, 10, 2, seed=13)
    print(model.predict(data))
