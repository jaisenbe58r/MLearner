"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import tensorflow as tf


def keras_checkpoint(model, optimizer, checkpoint_path="", max_to_keep=5):
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
    ckpt = tf.train.Checkpoint(Model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("The last checkpoint has been restored")

    return ckpt_manager


class MyCustomCallback(tf.keras.callbacks.Callback):
    """
    Keras Callback
    """
    def __init__(self, checkpoint_path, ckpt_manager):
        super(MyCustomCallback, self).__init__()

        self.checkpoint_path = checkpoint_path
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        print("Checkpoint guardado en {}.".format(self.checkpoint_path))


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Detener el entrenamiento cuando la perdida (loss) esta en su minimo,
    i.e. la perdida (loss) deja de disminuir.

    Parameters
    -----------
        patience: int
            Numero de epochs a esperar despues de que el min ha sido alcanzaado.
            Despues de este numero de no mejoras, el entrenamiento para.
    """
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()

        self.patience = patience

        # best_weights para almacenar los pesos en los cuales ocurre la perdida minima.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # El numero de epoch que ha esperado cuando la perdida ya no es minima.
        self.wait = 0
        # El epoch en el que en entrenamiento se detiene.
        self.stopped_epoch = 0
        # Initialize el best como infinito.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Guardar los mejores pesos si el resultado actual es mejor (menos).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restaurando los pesos del modelo del final de la mejor epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: Detencion anticipada' % (self.stopped_epoch + 1))


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Planificador de Learning rate que define el learning
    rate deacuerdo a lo programado.

    Parameters
    -----------
        schedule:
            una funcion que toma el indice del epoch
            (entero, indexado desde 0) y el learning rate actual
            como entradas y regresa un nuevo learning rate como salida (float).

    Examples:
    ---------
        ```python
        LR_SCHEDULE = [
            # (epoch a comenzar, learning rate) tupla
            (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
        ]

        def lr_schedule(epoch, lr):
        #Funcion de ayuda para recuperar el learning rate programado basado en la epoch.
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
        return lr

        model = get_model()
        _ = model.fit(x_train, y_train,
                batch_size=64,
                steps_per_epoch=5,
                epochs=15,
                verbose=0,
                callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])
        ```
    """
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Obtener el learning rate actua del optimizer del modelo.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Llamar la funcion schedule para obtener el learning rate programado.
        scheduled_lr = self.schedule(epoch, lr)
        # Definir el valor en el optimized antes de que la epoch comience
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))
