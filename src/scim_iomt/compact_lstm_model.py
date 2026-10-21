"""Compact float model used during the fixed-QAT search pipeline."""

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


def get_activation_function_by_name(name="exp"):
    """Map the config string to the std-head activation function."""
    if name == "exp":
        return tf.exp
    if name == "softplus":
        return tf.math.softplus
    if name == "relu":
        return tf.nn.relu
    raise ValueError("Activation function {} is not supported.".format(name))


def load(input_shape, output_shape, cfg):
    """Build the compact probabilistic LSTM architecture used before QAT."""
    nb_lstm_states = int(cfg["nb_lstm_states"])
    dense_units_1 = int(cfg.get("dense_units_1", 512))
    dense_units_2 = int(cfg.get("dense_units_2", 256))
    dropout_1 = float(cfg.get("dropout_1", 0.2))
    dropout_2 = float(cfg.get("dropout_2", 0.3))

    inputs = KL.Input(shape=input_shape)
    x = KL.LSTM(units=nb_lstm_states, unit_forget_bias=True)(inputs)

    if dense_units_1 > 0:
        x = KL.Dense(dense_units_1)(x)
        x = KL.Activation("relu")(x)
        if dropout_1 > 0.0:
            x = KL.Dropout(dropout_1)(x)

    if dense_units_2 > 0:
        x = KL.Dense(dense_units_2)(x)
        x = KL.Activation("relu")(x)
        if dropout_2 > 0.0:
            x = KL.Dropout(dropout_2)(x)

    if int(output_shape) == 2:
        mu = KL.Dense(1)(x)
        std = KL.Dense(1)(x)
        activation_fn = get_activation_function_by_name(cfg.get("activation_function", "exp"))
        std = KL.Activation(activation_fn, name="exponential_activation")(std)
        output = KL.Concatenate(axis=-1)([std, mu])
    else:
        output = KL.Dense(output_shape)(x)

    model = KM.Model(inputs=[inputs], outputs=[output])
    return model
