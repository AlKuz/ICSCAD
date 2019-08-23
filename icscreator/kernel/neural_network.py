"""
Class for creating neural networks
"""
import numpy as np
import os
import tensorflow as tf
from abc import abstractmethod
import tqdm
import json
from typing import List

from icscreator.kernel import Layer
from icscreator.kernel.visualization import VisualTool


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

    def predict_step(self, data_step: np.ndarray) -> np.ndarray:
        """
        Prediction only for one time step

        Args:
            data_step (np.ndarray): Data from the one time step

        Returns:
            np.ndarray: Result of the prediction for the one time step
        """
        return self._session.run(self._model_outputs, feed_dict={self._model_inputs: data_step})

    def predict(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Make prediction of neural network.

        Args:
            data (list): Make predictions for list of values.

        Returns (list): List of prediction results.
        """
        result = []
        for data_step in data:
            result_step = self.predict_step(data_step)
            result.append(result_step)
        return result

    def evaluate_step(self, input_step: np.ndarray, target_step: np.ndarray) -> float:
        """
        Evaluate only for one time step

        Args:
            input_step (np.ndarray): One time step data to feed neural network.
            target_step (np.ndarray): One time step data to compare network results.

        Returns (float): accuracy for one time step
        """
        result = self._session.run(self._model_loss, feed_dict={self._model_inputs: input_step,
                                                                self._model_targets: target_step})
        return float(result)

    def evaluate(self, input_data: List[np.ndarray], target_data: List[np.ndarray]) -> float:
        """
        Evaluate neural network.

        Args:
            input_data (np.ndarray): List of data to feed neural network.
            target_data (np.ndarray): List of data to compare network results.

        Returns (float): result accuracy
        """
        loss_result = []
        for i, t in zip(input_data, target_data):
            loss = self.evaluate_step(i, t)
            loss_result.append(loss)
        loss_result = float(np.mean(loss_result))
        return loss_result

    def fit_step(self, input_step: np.ndarray, target_step: np.ndarray):
        """
        Training step for one time step data

        Args:
            input_step (np.ndarray): Data to feed neural network
            target_step (np.ndarray): Data to compare network result
        """
        self._session.run([self._model_loss, self._model_optimizer],
                          feed_dict={self._model_targets: target_step,
                                     self._model_inputs: input_step})

    def train(self, input_data: List[np.ndarray], target_data: List[np.ndarray],
              folder: str = None, name: str = None, epochs=1000, early_stop=10, vis_tool: VisualTool = None):
        counter = 0
        with tqdm.tqdm(range(epochs)) as tqdm_epochs:
            for e in tqdm_epochs:
                for d in range(len(input_data)): self.fit_step(input_data[d], target_data[d])
                loss = self.evaluate(input_data, target_data)
                tqdm_epochs.set_description(desc='Epoch {}, loss={:.6f}'.format(e+1, loss))
                try:
                    if loss >= min_loss:
                        counter += 1
                    else:
                        min_loss = loss
                        counter = 0
                        self.save(folder, name)
                except NameError:
                    min_loss = loss

                if vis_tool is not None:
                    model_result = self.predict(input_data)
                    vis_tool.draw([target_data, model_result], [loss])

                if counter == early_stop:
                    break

    def save(self, folder: str, model_name: str = None):
        self._session.as_default()
        if model_name is None:
            model_name = self._name

        path = os.path.join(folder, model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        tf.compat.v1.train.Saver().save(self._session, os.path.join(path, 'model.ckpt'))

        object_fields = {k: v for k, v in self.__dict__.items() if k not in
                         ['_session', '_saver', '_model_inputs', '_model_outputs', '_model_targets', '_model_loss',
                          '_model_optimizer']}
        object_fields["input_name"] = self._model_inputs.name
        object_fields["output_name"] = self._model_outputs.name
        with open(os.path.join(folder, model_name, 'info.json'), 'w') as file:
            json.dump(object_fields, file)

    def load(self, folder: str):
        if self._model_inputs is not None or self._model_outputs is not None:
            print("Error: class contains model")
            return None

        with open(os.path.join(folder, 'info.json'), 'r') as file:
            fields: dict = json.load(file)

        self._session.as_default()
        saver = tf.train.Saver()
        saver.restore(self._session, os.path.join(folder, 'model.ckpt'))
        self._model_inputs = self._session.graph.get_tensor_by_name(fields["input_name"])
        self._model_outputs = self._session.graph.get_tensor_by_name(fields["output_name"])

        del fields["input_name"]
        del fields["output_name"]

        for key, value in fields.items():
            setattr(self, key, value)

    @property
    def session(self):
        return self._session


if __name__ == "__main__":
    from icscreator.prepared_models import ElmanNetwork

    FOLDER = '/home/alexander/Projects/ICSCreator/static/models'
    DATA_PATH = "/home/alexander/Projects/ICSCreator/static/data/Data_JC.csv"

    data_jc = np.genfromtxt(DATA_PATH, delimiter=',')

    fuel = np.expand_dims(data_jc[1::1000, 1], axis=-1) / 4.0
    freq = np.expand_dims(data_jc[1::1000, 2], axis=1) / 200000.0
    # temp = np.expand_dims(data_jc[1::100, 3] / 1000.0, axis=1)
    # freq_temp = np.concatenate([freq, temp], axis=1)

    vis_tool = VisualTool(titles=['Freq'],
                          x_info=['Time'],
                          y_info=['freq'],
                          legend=['Target', 'Model'],
                          ratio=9/16)

    engine = ElmanNetwork(1, 1, 1, seed=13, name='engine')
    engine.compile()
    engine.train(fuel, freq, folder=FOLDER, epochs=1000, vis_tool=vis_tool, early_stop=100)

