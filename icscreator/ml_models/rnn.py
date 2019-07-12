"""
Recurrent neural networks:

ElmanNetwork
"""
import tensorflow as tf


class ElmanNetwork(object):

    def __init__(self, inputs: int, hiddens: int, outputs: int, seed=None):
        self._inputs = inputs
        self._hiddens = hiddens
        self._outputs = outputs
        self._seed = seed
        self._graph = tf.Graph()
        self._model_inputs = None
        self._model_outputs = None
        self._create_model()

    def _create_model(self):
        with self._graph.as_default():
            self._model_inputs = tf.compat.v1.placeholder(tf.float32, [1, self._inputs], 'inputs')
            state = tf.Variable(tf.zeros([1, self._hiddens]), trainable=False, name='state')

            with tf.name_scope('hidden'):
                wh = tf.Variable(tf.random.uniform([self._inputs, self._hiddens], -1, 1, seed=self._seed, name='wh'))
                ws = tf.Variable(tf.random.uniform([self._hiddens, self._hiddens], -1, 1, seed=self._seed, name='ws'))
                bh = tf.Variable(tf.random.uniform([self._hiddens], -1, 1, seed=self._seed, name='bh'))
                hiddens = tf.assign(state, tf.sigmoid(tf.matmul(self._model_inputs, wh) + tf.matmul(state, ws) + bh))

            with tf.name_scope('output'):
                wo = tf.Variable(tf.random.uniform([self._hiddens, self._outputs], -1, 1, seed=self._seed, name='wo'))
                bo = tf.Variable(tf.random.uniform([self._outputs], -1, 1, seed=self._seed, name='bo'))
                self._model_outputs = tf.sigmoid(tf.matmul(hiddens, wo) + bo, name='outputs')

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
