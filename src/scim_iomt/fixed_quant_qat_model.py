#!/usr/bin/env python3
"""Fixed-range activation and effective signed INT4-style QAT model for compact search."""

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


try:
    _AbstractRNNCellBase = KL.AbstractRNNCell
except AttributeError:
    # Newer TensorFlow/Keras builds may no longer expose AbstractRNNCell.
    # A Layer subclass with state_size/output_size is still accepted by KL.RNN.
    _AbstractRNNCellBase = KL.Layer


def _ste_clip_and_quantize_unsigned(x, min_value: float, max_value: float, num_bits: int = 8):
    """Unsigned fake quantizer with straight-through gradients."""
    min_t = tf.cast(min_value, tf.float32)
    max_t = tf.cast(max_value, tf.float32)
    qmax = float((1 << num_bits) - 1)
    scale = tf.maximum((max_t - min_t) / qmax, tf.constant(1e-8, dtype=tf.float32))
    clipped = tf.clip_by_value(tf.cast(x, tf.float32), min_t, max_t)
    q = tf.round((clipped - min_t) / scale)
    q = tf.clip_by_value(q, 0.0, qmax)
    dequant = q * scale + min_t
    return clipped + tf.stop_gradient(dequant - clipped)


def _ste_clip_and_quantize_signed(x, range_abs: float, num_bits: int = 8):
    """Signed fake quantizer with straight-through gradients."""
    range_t = tf.cast(range_abs, tf.float32)
    qmax = float((1 << (num_bits - 1)) - 1)
    scale = tf.maximum(range_t / qmax, tf.constant(1e-8, dtype=tf.float32))
    clipped = tf.clip_by_value(tf.cast(x, tf.float32), -range_t, range_t)
    q = tf.round(clipped / scale)
    q = tf.clip_by_value(q, -qmax, qmax)
    dequant = q * scale
    return clipped + tf.stop_gradient(dequant - clipped)


def _ste_weight_quant_per_output_channel(weights, cell_bits: int = 3):
    """Per-output-channel fake quantization; cell_bits=3 gives signed levels [-7, 7]."""
    qmax = float((1 << cell_bits) - 1)
    x = tf.cast(weights, tf.float32)
    max_abs = tf.reduce_max(tf.abs(x), axis=0, keepdims=True)
    max_abs = tf.maximum(max_abs, tf.constant(1e-8, dtype=tf.float32))
    scale = max_abs / qmax
    clipped = tf.clip_by_value(x, -max_abs, max_abs)
    q = tf.round(clipped / scale)
    q = tf.clip_by_value(q, -qmax, qmax)
    dequant = q * scale
    return clipped + tf.stop_gradient(dequant - clipped)


class FixedQuantDense(KL.Layer):
    """Dense layer that applies activation and weight fake quantization in call()."""
    def __init__(
        self,
        units: int,
        *,
        input_signed: bool,
        input_min=None,
        input_max=None,
        input_abs=None,
        weight_bits: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.input_signed = bool(input_signed)
        self.input_min = input_min
        self.input_max = input_max
        self.input_abs = input_abs
        self.weight_bits = int(weight_bits)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        if self.input_signed:
            x_q = _ste_clip_and_quantize_signed(inputs, self.input_abs, num_bits=8)
        else:
            x_q = _ste_clip_and_quantize_unsigned(
                inputs,
                self.input_min,
                self.input_max,
                num_bits=8,
            )
        w_q = _ste_weight_quant_per_output_channel(self.kernel, cell_bits=self.weight_bits)
        return tf.matmul(x_q, w_q) + self.bias


class FixedQuantLSTMCell(_AbstractRNNCellBase):
    """LSTM cell whose inputs, hidden state, and weights are fake-quantized."""
    def __init__(
        self,
        units: int,
        *,
        input_min: float,
        input_max: float,
        hidden_abs: float,
        weight_bits: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.hidden_abs = float(hidden_abs)
        self.weight_bits = int(weight_bits)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units * 4),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, self.units * 4),
            initializer="orthogonal",
            trainable=True,
        )
        forget_bias = tf.concat(
            [
                tf.zeros((self.units,), dtype=tf.float32),
                tf.ones((self.units,), dtype=tf.float32),
                tf.zeros((self.units * 2,), dtype=tf.float32),
            ],
            axis=0,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units * 4,),
            initializer=tf.keras.initializers.Constant(forget_bias.numpy()),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_prev, c_prev = states
        x_q = _ste_clip_and_quantize_unsigned(inputs, self.input_min, self.input_max, num_bits=8)
        h_q = _ste_clip_and_quantize_signed(h_prev, self.hidden_abs, num_bits=8)
        kernel_q = _ste_weight_quant_per_output_channel(self.kernel, cell_bits=self.weight_bits)
        recurrent_q = _ste_weight_quant_per_output_channel(
            self.recurrent_kernel,
            cell_bits=self.weight_bits,
        )

        z = tf.matmul(x_q, kernel_q) + tf.matmul(h_q, recurrent_q) + self.bias
        i, f, g, o = tf.split(z, num_or_size_splits=4, axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        g = tf.tanh(g)
        o = tf.sigmoid(o)
        c = f * c_prev + i * g
        h = o * tf.tanh(c)
        h_out = _ste_clip_and_quantize_signed(h, self.hidden_abs, num_bits=8)
        return h_out, [h_out, c]


def get_activation_function_by_name(name="exp"):
    """Map the std-head activation string to a TensorFlow callable."""
    if name == "exp":
        return tf.exp
    if name == "softplus":
        return tf.math.softplus
    if name == "relu":
        return tf.nn.relu
    raise ValueError("Activation function {} is not supported.".format(name))


def load(input_shape, output_shape, cfg):
    """Build the end-to-end QAT model used for the compact deployment target."""
    nb_lstm_states = int(cfg["nb_lstm_states"])
    dense_units_1 = int(cfg.get("dense_units_1", 512))
    dense_units_2 = int(cfg.get("dense_units_2", 256))
    dropout_1 = float(cfg.get("dropout_1", 0.2))
    dropout_2 = float(cfg.get("dropout_2", 0.3))

    quant_input_min = float(cfg["quant_input_min"])
    quant_input_max = float(cfg["quant_input_max"])
    quant_hidden_abs = float(cfg["quant_hidden_abs"])
    quant_relu1_max = float(cfg["quant_relu1_max"])
    quant_relu2_max = float(cfg["quant_relu2_max"])
    quant_weight_bits = int(cfg.get("quant_weight_bits", 3))

    inputs = KL.Input(shape=input_shape)
    x = KL.RNN(
        FixedQuantLSTMCell(
            units=nb_lstm_states,
            input_min=quant_input_min,
            input_max=quant_input_max,
            hidden_abs=quant_hidden_abs,
            weight_bits=quant_weight_bits,
            name="qat_lstm_cell",
        ),
        return_sequences=False,
        name="qat_lstm",
    )(inputs)

    if dense_units_1 > 0:
        x = FixedQuantDense(
            dense_units_1,
            input_signed=True,
            input_abs=quant_hidden_abs,
            weight_bits=quant_weight_bits,
            name="qat_dense1",
        )(x)
        x = KL.Activation("relu", name="qat_relu1")(x)
        if dropout_1 > 0.0:
            x = KL.Dropout(dropout_1, name="qat_dropout1")(x)

    if dense_units_2 > 0:
        x = FixedQuantDense(
            dense_units_2,
            input_signed=False,
            input_min=0.0,
            input_max=quant_relu1_max,
            weight_bits=quant_weight_bits,
            name="qat_dense2",
        )(x)
        x = KL.Activation("relu", name="qat_relu2")(x)
        if dropout_2 > 0.0:
            x = KL.Dropout(dropout_2, name="qat_dropout2")(x)

    std = FixedQuantDense(
        1,
        input_signed=False,
        input_min=0.0,
        input_max=quant_relu2_max,
        weight_bits=quant_weight_bits,
        name="qat_std_head",
    )(x)
    mu = FixedQuantDense(
        1,
        input_signed=False,
        input_min=0.0,
        input_max=quant_relu2_max,
        weight_bits=quant_weight_bits,
        name="qat_mu_head",
    )(x)
    activation_fn = get_activation_function_by_name(cfg.get("activation_function", "exp"))
    std = KL.Activation(activation_fn, name="exponential_activation")(std)
    output = KL.Concatenate(axis=-1)([std, mu])

    if int(output_shape) != 2:
        raise ValueError("FixedQuant QAT model currently expects probabilistic output_shape=2")
    return KM.Model(inputs=[inputs], outputs=[output])


def copy_weights_from_float_model(float_model, quant_model):
    """Initialize the QAT model from a pretrained float checkpoint."""
    float_lstm = float_model.get_layer("lstm")
    quant_lstm = quant_model.get_layer("qat_lstm").cell
    quant_lstm.kernel.assign(float_lstm.weights[0])
    quant_lstm.recurrent_kernel.assign(float_lstm.weights[1])
    quant_lstm.bias.assign(float_lstm.weights[2])

    quant_model.get_layer("qat_dense1").kernel.assign(float_model.get_layer("dense").kernel)
    quant_model.get_layer("qat_dense1").bias.assign(float_model.get_layer("dense").bias)
    quant_model.get_layer("qat_dense2").kernel.assign(float_model.get_layer("dense_1").kernel)
    quant_model.get_layer("qat_dense2").bias.assign(float_model.get_layer("dense_1").bias)
    quant_model.get_layer("qat_std_head").kernel.assign(float_model.get_layer("dense_3").kernel)
    quant_model.get_layer("qat_std_head").bias.assign(float_model.get_layer("dense_3").bias)
    quant_model.get_layer("qat_mu_head").kernel.assign(float_model.get_layer("dense_2").kernel)
    quant_model.get_layer("qat_mu_head").bias.assign(float_model.get_layer("dense_2").bias)
    return quant_model
