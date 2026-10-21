"""Negative log-likelihood loss for mean/std glucose prediction."""

import math

import tensorflow as tf


def load():
    """Expose the loss function through the config-driven loader interface."""
    return tf_nll


def tf_nll(y_true, y_pred):
    """Gaussian NLL where the model outputs `[std, mean]` for each sample."""
    y_std = tf.maximum(y_pred[:, :1], tf.keras.backend.epsilon())
    y_mean = y_pred[:, 1:]
    z = (y_true - y_mean) / y_std
    log_prob = -0.5 * tf.square(z) - tf.math.log(y_std) - 0.5 * math.log(2.0 * math.pi)
    return tf.reduce_mean(-log_prob)
