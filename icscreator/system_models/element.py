"""
System element is interface between ML models and system
"""
from keras.models import Model as KModel
from keras import backend as K

from icscreator.system_models.model import Model


class Element(Model):

    """
    Class-interface for ML models in system.
    """

    def __init__(self, model: KModel):
        """
        Initialization of Element.
        :param model:
        """
        self._model = model
        self._sess = K.get_session()

    def predict(self, input_data):
        self._model.predict(input_data)

    def fit(self, train_data, test_data):
        pass

    def get_input(self):
        return self._model.input

    def get_output(self):
        return self._model.output

    def get_session(self):
        return self._sess

    def set_session(self, session):
        self._sess = session
