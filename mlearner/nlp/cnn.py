"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import tensorflow as tf
from tensorflow.keras import layers

"""
Based on https://www.udemy.com/course/procesamiento-del-lenguaje-natural/learn/lecture/21502260#overview

Author: Juan Gabriel Gomila
Course: "Procesamiento del Lenguaje Natural Moderno en Python.

Original Text: https://arxiv.org/pdf/1404.2188.pdf

"""


class DCNN(tf.keras.Model):
    """
    The DCNN class corresponds to the Neural Convolution Network algorithm
    for Natural Language Processing.

    Parameters
    ----------
        vocab_size:
            Vocabulary size of the algorithm input text.

        emb_dim : int
            Embedding size.

        nb_filters : int
            Filter size for each layer Conv1D.

        FFN_units : int
            Units for dense layer.

        nb_classes : int
            Numbers of final categories.

        dropout_rate : float
            Dropout parameter.

        training : bool
            Trainning process activated.

        name : str
            Custom Model Name.

        weights_path: str
            Path load weight model.

    Attributes
    -----------
        embedding : tf.keras.layers.Embedding
            Embedding layer for input vocabulary.

        bigram : tf.keras.layers.Conv1D
            1D convolution layer, for two letters in a row.

        trigram : tf.keras.layers.Conv1D
            1D convolution layer, for three letters in a row.

        fourgram : tf.keras.layers.Conv1D
            1D convolution layer, for four letters in a row.

        pool : tf.keras.layers.GlobalMaxPool1D
            Max pooling operation for 1D temporal data.

        dense_1 : tf.keras.layers.Dense
            Regular densely-connected NN layer, concatenate 1D Convolutions.

        last_dense : tf.keras.layers.Dense
            Regular densely-connected NN layer, final decision.

        dropout : tf.keras.layers.Dropout
            Applies Dropout to dense_1.

    Examples:
    ---------
        ```python
        VOCAB_SIZE = tokenizer.vocab_size # 65540
        EMB_DIM = 200
        NB_FILTERS = 100
        FFN_UNITS = 256
        NB_CLASSES = 2#len(set(train_labels))
        DROPOUT_RATE = 0.2
        BATCH_SIZE = 32
        NB_EPOCHS = 5

        Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

        if NB_CLASSES == 2:
            Dcnn.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=["accuracy"])
        else:
            Dcnn.compile(loss="sparse_categorical_crossentropy",
                        optimizer="adam",
                        metrics=["sparse_categorical_accuracy"])

        # Entrenamiento
        Dcnn.fit(train_inputs,
                train_labels,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCHS)

        # Evaluation
        results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
        print(results)
        ```
    """
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 weights_path=None,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)

        self.weights_path = weights_path

        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        # No tenemos variable de entrenamiento
        # asi que podemos usar la misma capa
        # para cada paso de pooling
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        """
        Calling the build function of the model.

        Parameters
        ----------
            inputs: Tensor.
                Input Tensor.

            Training : bool
                Trainning process activated.

        Returns:
            output: Tensor.
                Output Tensor.
        """
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1)  # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        # Load weights.
        if self.weights_path is not None:
            output.load_weights(self.weights_path)

        return output
