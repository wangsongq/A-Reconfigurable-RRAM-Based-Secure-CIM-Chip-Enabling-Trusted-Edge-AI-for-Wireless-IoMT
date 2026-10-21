"""Optimizer factory used by the experiment configs."""

import tensorflow as tf

def load(cfg):
    """Build an Adam optimizer from a small configuration dictionary."""
    learning_rate = float(cfg['learning_rate'])
    return tf.keras.optimizers.Adam(learning_rate)
