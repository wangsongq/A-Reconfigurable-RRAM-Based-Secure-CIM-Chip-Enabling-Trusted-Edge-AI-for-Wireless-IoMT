"""Keras training helpers for the released float-model pipeline."""

import os

import tensorflow as tf


class TrainingLogger(tf.keras.callbacks.Callback):
    """Track the epoch with the best validation loss for easier reporting."""

    def __init__(self):
        super().__init__()
        self.best_epoch = -1
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        print(f"best_val_loss={self.best_val_loss:.4f} at epoch={self.best_epoch + 1}")


def train(
    model,
    x_train,
    y_train,
    x_valid,
    y_valid,
    batch_size,
    epochs,
    patience,
    shuffle,
    artifacts_path,
):
    """Train a Keras model with early stopping, LR scheduling, and checkpointing."""
    logger_cb = TrainingLogger()

    # Keep all generated outputs under one directory so the repository can
    # ignore them cleanly and users can archive a full run in one place.
    os.makedirs(artifacts_path, exist_ok=True)
    model_filepath = os.path.join(artifacts_path, "model.hdf5")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience,
            min_lr=1e-6,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=artifacts_path),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch",
            verbose=1,
        ),
        logger_cb,
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=callbacks,
    )

    print(f"best_model_path={model_filepath}")
    return model
