"""Module for different prepared neural network models"""

from icscreator.kernel.layers import Input, Dense, SRNN
import tensorflow as tf


def elman_network(inputs, hiddens, outputs, activation: str = 'sigmoid') -> tf.keras.Model:
    inputs = (inputs,) if isinstance(inputs, int) else inputs
    hiddens = (hiddens,) if isinstance(hiddens, int) else hiddens
    outputs = (outputs,) if isinstance(outputs, int) else outputs

    model_input = Input(shape=inputs)()
    model = SRNN(shape=hiddens, activation='relu')(model_input)
    model = Dense(shape=outputs, activation=activation)(model)
    model = tf.keras.Model(model_input, model)
    return model


if __name__ == "__main__":
    elman_model = elman_network(inputs=5, hiddens=(10, 10, 10), outputs=2)
    elman_model.summary()