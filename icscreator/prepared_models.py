"""Module for different prepared neural network models"""

from icscreator.kernel import Layer, NeuralNetwork


class ElmanNetwork(NeuralNetwork):
    """Elman network"""

    def __init__(self, inputs: int, hiddens: int, outputs: int, seed=None, name: str = None):
        self._inputs = inputs
        self._hiddens = hiddens
        self._outputs = outputs
        Layer._seed = seed
        super().__init__(name='elman_network' if name is None else name)

    def _create_model(self):
        self._model_inputs = Layer.input(self._inputs)
        state = Layer.state(self._hiddens)
        hiddens = Layer.dense(Layer.concat([self._model_inputs, state], axis=0), self._hiddens, name='hidden')
        hiddens = Layer.assign(target=state, value=hiddens)
        self._model_outputs = Layer.dense(hiddens, self._outputs, name='output')
