"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import tensorflow as tf


def keras_checkpoint(model, checkpoint_path="", max_to_keep=5):
    """
    Manantiene una save_counter para numerar los puestos de control.

    Parameters
    ----------
        model: tf.keras.Model.
            Input model

        checkpoint_path: str
            Input model

        max_to_keep: int
            Maximo numero de checkpoints guardados

    Returns:
    --------
        ckpt_manager: tf.train.Checkpoint.
            Clase Checkpoint
    """
    ckpt = tf.train.Checkpoint(Model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("The last checkpoint has been restored")

    return ckpt_manager
