"""
Interface for all models
"""
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Main class. Everything is model.
    """

    @abstractmethod
    def predict(self, input_data):
        """
        Calculate model prediction.
        :param input_data:
        :return:
        """

    @abstractmethod
    def fit(self, train_data, test_data):
        pass

    @abstractmethod
    def set_session(self, session):
        pass

    @abstractmethod
    def get_session(self):
        pass

    @abstractmethod
    def get_input(self):
        """
        Input layer of the Keras model.
        :return: Tensor
        """

    @abstractmethod
    def get_output(self):
        """
        Output layer of the Keras model.
        :return: Tensor
        """
