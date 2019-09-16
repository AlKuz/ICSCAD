import tensorflow as tf
import numpy as np
from abc import abstractmethod
import os
from icscreator.kernel.layers import Input
from icscreator.kernel.visualization import VisualTool, EmptyVisualTool
from tqdm import tqdm


class Model(object):

    _OPTIMIZERS = {
        'adam': tf.train.AdamOptimizer,
        'gradient': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'rms': tf.train.RMSPropOptimizer
    }
    _LOSSES = {
        'mse': tf.losses.mean_squared_error,
        'sigmoid_cross_entropy': tf.losses.sigmoid_cross_entropy,
        'softmax_cross_entropy': tf.losses.softmax_cross_entropy
    }

    def __init__(self, name: str = None):
        self._name = self.__class__.__name__ if name is None else name
        self._session = tf.Session()
        self._input = None
        self._output = None
        self._target = None
        self._optimizer = None
        self._loss = None
        self._saver = None

    @abstractmethod
    def _build_model(self) -> (tf.Tensor, tf.Tensor):
        """
        Create custom model graph

        Returns:
            (tf.Tensor, tf.Tensor): (model_input, model_output)
        """
        raise NotImplemented("Method '_build_model' is not implemented is {}".format(self.__class__.__name__))

    def save(self, path: str):
        folder = os.path.join(path, self._name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        model_path = os.path.join(folder, self._name + '.ckpt')
        self._saver.save(self._session, model_path)
        print("Model was saved to", folder)

    def load(self, folder: str):
        model_path = os.path.join(folder, self._name + '.ckpt')
        self._session.as_default()
        self._input, self._output = self._build_model()
        self._saver = tf.train.Saver()
        self._saver.restore(self._session, model_path)
        print("Model was loaded from", folder)

    def compile(self, loss: str, optimizer: str, optimizer_parameters: dict):
        """
        Compile model graph

        Args:
            loss (str): Cost function (https://www.tensorflow.org/api_docs/python/tf/losses)
            optimizer (str): Optimizer to update model weights (https://www.tensorflow.org/api_docs/python/tf/train)
            optimizer_parameters (dict):
                adam: {learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08}
                gradient: {learning_rate}
                momentum: {learning_rate, momentum, use_nesterov=False}
                rms: {learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10}
        """
        assert optimizer in self._OPTIMIZERS, "Use optimizers from the list: {}".format(list(self._OPTIMIZERS.keys()))
        assert loss in self._LOSSES, "Use losses from the list: {}".format(list(self._LOSSES.keys()))
        self._session.as_default()
        self._input, self._output = self._build_model()
        self._target = Input(shape=self._output.shape.as_list(), name='Target')
        self._loss = self._LOSSES[loss](self._target, self._output)
        self._optimizer = self._OPTIMIZERS[optimizer](**optimizer_parameters).minimize(self._loss)
        self._session.run(tf.compat.v1.global_variables_initializer())
        self._saver = tf.train.Saver()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result = []
        for data in input_data:
            r = self._session.run(self._output, feed_dict={self._input: data})
            result.append(r)
        result = np.array(result)
        return result

    def fit(self, input_data, target_data, epochs: int = 1000, vizualizer: VisualTool = EmptyVisualTool,
            model_path=None):
        self._session.as_default()
        mean_loss = None
        mean_error = None
        for e in range(epochs):
            losses = []
            for input_sample, target_sample in tqdm(zip(input_data, target_data),
                                                    desc="Epoch {}: loss = {}".format(e, mean_loss)):
                loss, _ = self._session.run([self._loss, self._optimizer], feed_dict={
                    self._input: input_sample,
                    self._target: target_sample}
                )
                losses.append(loss)
            mean_loss = sum(losses) / len(losses)
            model_result = self.predict(input_data)
            errors = np.mean(np.abs(target_data - model_result) / target_data * 100, axis=0)
            if mean_error is None:
                mean_error = np.mean(errors)
            elif np.mean(errors) < mean_error and model_path is not None:
                self.save(model_path)
                mean_error = np.mean(errors)
            print("Parameters errors: ", errors)
            vizualizer.draw([model_result, target_data], list(errors))