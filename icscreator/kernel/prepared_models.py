"""Module for different prepared neural network models"""

from icscreator.kernel.layers import Input, Dense, SRNN
import tensorflow as tf
from typing import Tuple


def multilayer_srnn(*layer_sizes, activations: Tuple[str] = ('sigmoid',), name="multilayer_srnn") -> tf.keras.Model:
    assert len(layer_sizes) > 1, "Network should have more than one layer"
    if len(activations) == 1:
        activations = activations * (len(layer_sizes) - 1)
    else:
        assert len(layer_sizes) - 1 == len(activations)

    model_input = tf.keras.layers.Input([layer_sizes[0]])
    model = tf.keras.layers.LSTM(layer_sizes[1], stateful=True)(model_input)
    for layer in layer_sizes[2:]:
        model = tf.keras.layers.LSTM(layer, stateful=True)(model)
    model = tf.keras.Model(model_input, model, name=name)

    # model = SRNN(input_shape=model_input.shape.as_list(), output_shape=layer_sizes[1], activation=activations[0],
    #              name='srnn_0')(model_input)
    # for e, (layer, activation) in enumerate(zip(layer_sizes[2:], activations[1:])):
    #     model = SRNN(input_shape=model.shape.as_list(), output_shape=layer, activation=activation,
    #                  name='srnn_{}'.format(e + 1))(model)
    # model = tf.keras.Model(model_input, model, name=name)
    return model


def multilayer_perceptron(*layer_sizes, activations: Tuple[str] = ('sigmoid',)) -> tf.keras.Model:
    assert len(layer_sizes) > 1, "Network should have more than one layer"
    if len(activations) == 1:
        activations = activations * (len(layer_sizes) - 1)
    else:
        assert len(layer_sizes) - 1 == len(activations)

    model_input = Input(layer_sizes[0])
    model = Dense(shape=layer_sizes[1], activation=activations[0], name='dense_0')(model_input)
    for e, (layer, activation) in enumerate(zip(layer_sizes[2:], activations[1:])):
        model = Dense(shape=layer, activation=activation, name='dense_{}'.format(e+1))(model)
    model = tf.keras.Model(model_input, model)
    return model


def elman_network(inputs, hiddens, outputs, activation: str = 'sigmoid') -> tf.keras.Model:
    inputs = (inputs,) if isinstance(inputs, int) else inputs
    hiddens = (hiddens,) if isinstance(hiddens, int) else hiddens
    outputs = (outputs,) if isinstance(outputs, int) else outputs

    model_input = Input(shape=inputs)
    model = SRNN(output_shape=hiddens, activation='relu')(model_input)
    model = Dense(shape=outputs, activation=activation)(model)
    model = tf.keras.Model(model_input, model)
    return model


if __name__ == "__main__":
    elman_model = elman_network(inputs=5, hiddens=(10, 10, 10), outputs=2)
    elman_model.summary()

    multilayer_model = multilayer_perceptron((3, 4, 5), (6, 7), (8,), activations=('tanh',))
    multilayer_model.summary()

    multilayer_srnn_model = multilayer_srnn((3, 4, 5), (6, 7), (8,), activations=('tanh',))
    multilayer_srnn_model.summary()
