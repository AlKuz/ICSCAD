"""
Class for creating neural networks
"""
import numpy as np
import os
import tensorflow as tf
from abc import abstractmethod
import tqdm
import json

from icscreator.kernel import Layer


class NeuralNetwork(object):
    """Base class of neural networks"""

    def __init__(self, name='neural_network'):
        """
        Initialization of base network class.
        """
        self._name = name
        self._session = tf.Session()
        self._model_inputs = None
        self._model_outputs = None
        self._model_targets = None
        self._model_loss = None
        self._model_optimizer = None

    def compile(self,
                loss=tf.losses.mean_squared_error,
                optimizer=tf.train.AdamOptimizer,
                **optimizer_settings) -> None:

        self._session.as_default()
        with tf.name_scope(self._name):
            self._create_model()

        assert isinstance(self._model_inputs, tf.Tensor)
        assert isinstance(self._model_outputs, tf.Tensor)
        self._model_targets = Layer.input(self._model_outputs.shape[0], name='targets')

        self._model_loss = loss(self._model_targets, self._model_outputs)
        self._model_optimizer = optimizer(**optimizer_settings).minimize(self._model_loss)
        self._session.run(tf.initialize_all_variables())

    @abstractmethod
    def _create_model(self):
        """
        Method for model creating.

        Use:
            self._model_inputs for input placeholder
            self._model_outputs for output tensor
        """
        pass

    def predict(self, data):
        """
        Make prediction of neural network.

        Args:
            data (list): Make predictions for list of values.

        Returns (list): List of prediction results.
        """
        result = []
        for i in data:
            r = self._session.run(self._model_outputs, feed_dict={self._model_inputs: i})
            r = list(r) if len(r) > 1 else float(r)
            result.append(r)
        return result

    def evaluate(self, input_data: list, target_data: list) -> float:
        """
        Evaluate neural network.

        Args:
            input_data (list): List of data to feed neural network.
            target_data (list): List of data to compare network results.

        Returns (float):
        """
        loss_result = []
        for i, t in zip(input_data, target_data):
            r = self._session.run(self._model_loss, feed_dict={self._model_inputs: i,
                                                               self._model_targets: t})
            loss_result.append(r)
        loss_result = float(np.mean(loss_result))
        return loss_result

    def train(self, input_data: list, target_data: list, folder: str = None, name: str = None,
              epochs=1000, early_stop=10):
        counter = 0
        for i in range(epochs):
            with tqdm.tqdm(range(len(input_data))) as tqdm_generator:
                for d in tqdm_generator:
                    loss, _ = self._session.run([self._model_loss, self._model_optimizer], feed_dict={
                        self._model_targets: target_data[d],
                        self._model_inputs: input_data[d]
                    })
                    tqdm_generator.set_description(desc='Epoch {}, loss={:.4f}'.format(i+1, loss))
                try:
                    if loss < min_loss:
                        min_loss = loss
                        counter = 0
                        self.save(folder, name)
                    else:
                        counter += 1
                except NameError:
                    min_loss = loss

                if counter == early_stop:
                    break

    def save(self, folder: str, model_name: str = None):
        self._session.as_default()
        if model_name is None:
            model_name = self._name
        model = tf.keras.models.Model(self._model_inputs, self._model_outputs)
        model.save(os.path.join(folder, model_name, 'model.hdf5'), include_optimizer=False)

        object_fields = {k: v for k, v in self.__dict__.items() if k not in
                         ['_session', '_model_inputs', '_model_outputs', '_model_targets', '_model_loss',
                          '_model_optimizer']}
        with open(os.path.join(folder, model_name, 'info.json'), 'w') as file:
            json.dump(object_fields, file)

    @property
    def session(self):
        return self._session


def load(folder: str, model_name: str) -> NeuralNetwork:
    with open(os.path.join(folder, model_name, 'info.json'), 'r') as file:
        fields: dict = json.load(file)

    model: tf.keras.models.Model = tf.keras.models.load_model(os.path.join(folder, model_name, 'model.hdf5'),
                                                              compile=False)
    fields['_model_inputs'] = model.input
    fields['_model_outputs'] = model.output

    neural_network = NeuralNetwork()
    neural_network.session.as_default()
    for key, value in fields.items():
        setattr(neural_network, key, value)

    return neural_network


if __name__ == "__main__":
    from icscreator.prepared_models import ElmanNetwork

    FOLDER = os.path.join(os.getcwd(), 'static', 'models')

    data = [[1, 2, 3]] * 100
    target = [[0.1, 0.7]] * 100
    model1 = ElmanNetwork(3, 10, 2, seed=13)
    model1.train(data, target, folder=FOLDER)
    model1.compile()

    model2 = load(FOLDER, 'elman_network')
