"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers

"""
Based on https://www.udemy.com/course/procesamiento-del-lenguaje-natural/learn/lecture/21502260#overview

Author: Juan Gabriel Gomila
Course: "Procesamiento del Lenguaje Natural Moderno en Python.

Original Text: https://arxiv.org/pdf/1404.2188.pdf

"""

"""
Build model layers.
"""


class Widening(layers.Layer):
    """
    Widening Layer: Padding for an extended 2D convolution

    input: (batches, seq_len, d_model, channels)

    Parameters
    ----------
        padding_size: int
            Padding size of the algorithm input text.

        padding_value: float
            Padding value.
    """
    def __init__(self, padding_size, padding_value=0., axis=1):
        super(Widening, self).__init__()
        self.padding_size = padding_size
        self.padding_value = padding_value
        self.axis = axis

    def build(self, input_shape):
        paddings_arr = np.array([[0, 0]]*len(input_shape))
        paddings_arr[self.axis, :] = [self.padding_size, self.padding_size]
        self.paddings = tf.convert_to_tensor(paddings_arr, dtype="int32")

    def call(self, inputs):
        return tf.pad(inputs,
                      self.paddings,
                      constant_values=self.padding_value)


class MyConv2D(layers.Layer):
    """
    MyConv2D Layer: 2D convolution

    Parameters
    ----------
        nb_filters: int
            Number of filters.

        conv_width: float
            Convolution 2D size

        padding: int
            Padding size of the algorithm input text.

        emb_dim: float
            Embeding size of the algorithm input text.
    """
    def __init__(self,
                 nb_filters,
                 conv_width,
                 padding,
                 emb_dim):
        super(MyConv2D, self).__init__()
        self.emb_dim = emb_dim
        self.conv_per_col = [layers.Conv2D(filters=nb_filters,
                                           kernel_size=[conv_width, 1],
                                           padding="valid",
                                           activation="tanh")
                             for _ in range(emb_dim)]

    def call(self, inputs):
        convolutions = []
        for i in range(self.emb_dim):
            convolutions.append(self.conv_per_col[i](
                tf.expand_dims(inputs[:, :, i, :], axis=-2)))
        return tf.concat(convolutions, axis=-2)


# k-max pooling
class KMaxPooling(layers.Layer):
    """
    KMaxPooling Layer: a pooling operation that is a generalisation
    of the max pooling over the time dimension used in the Max-TDNN
    sentence model and different from the local max pooling operations
    applied in a convolutional network for object recognition
    (LeCun et al., 1998).

    Parameters
    ----------
        ktop: int
            is the fixed pooling parameter for the
            topmost convolutional layer (Sect. 3.2 - https://arxiv.org/pdf/1404.2188.pdf)
            For instance, in a network with three convolutional layers and ktop = 3

        L: int
            L is the total number of convolutional layers in the network

        l: int
            l is the number of the current convolutional
            layer to which the pooling is applied
    """
    def __init__(self, ktop=4, L=None, l=None):
        super(KMaxPooling, self).__init__()
        self.ktop = ktop
        self.L = L
        self.l = l

    def build(self, input_shape):
        s = input_shape[1]
        if self.L is None or self.l is None or s is None:
            self.k = self.ktop
        else:
            self.k = max(self.ktop, math.ceil((self.L-self.l)/self.L*s))

    def call(self, inputs):
        inputs_trans = tf.transpose(inputs, [0, 3, 2, 1])
        inputs_trans_kmax = tf.math.top_k(inputs_trans, self.k).values
        inputs_kmax = tf.transpose(inputs_trans_kmax, [0, 3, 2, 1])
        return inputs_kmax


class Folding(layers.Layer):
    """
    Folding Layer: In the formulation of the network so far, feature
    detectors applied to an individual row of the sentence matrix s can have many orders and create
    complex dependencies across the same rows in
    multiple feature maps. Feature detectors in different rows, however, are independent of each other
    until the top fully connected layer. Full dependence between different rows could be achieved
    by making M in Eq. 5 a full matrix instead of
    a sparse matrix of diagonals. Here we explore a
    simpler method called folding that does not introduce any additional parameters.
    After a convolutional layer and before (dynamic) k-max pooling, one just
    sums every two rows in a feature map component-wise.
    For a map of d rows, folding returns a map of d/2 rows, thus halving the size of
    the representation. With a folding layer, a feature
    detector of the i-th order depends now on two rows
    of feature values in the lower maps of or

     https://arxiv.org/pdf/1404.2188.pdf

    """
    def __init__(self):
        super(Folding, self).__init__()

    def call(self, inputs):
        folded_inputs = tf.math.add_n(
            [inputs[:, :, 0::2, :], inputs[:, :, 1::2, :]]
        ) / 2
        return folded_inputs


"""
Buid CNN Model
"""


class DCNN_Advanced(tf.keras.Model):
    """
    A Convolutional Neural Network for Modelling Sentences

    The ability to accurately represent sentences is central
    to language understanding. We describe a convolutional
    architecture dubbed the Dynamic Convolutional
    Neural Network (DCNN) that we adopt
    for the semantic modelling of sentences.
    The network uses Dynamic k-Max Pooling, a global pooling
    operation over linear sequences. The network handles input
    sentences of varying length and induces
    a feature graph over the sentence that is
    capable of explicitly capturing short and
    long-range relations. The network does
    not rely on a parse tree and is easily applicable to any language.
    We test the DCNN in four experiments: small scale
    binary and multi-class sentiment prediction, six-way question classification and
    Twitter sentiment prediction by distant supervision. The network achieves excellent
    performance in the first three tasks and a
    greater than 25% error reduction in the last
    task with respect to the strongest baseline

    Parameters
    ----------
        vocab_size:
            Vocabulary size of the algorithm input text.

        emb_dim : int
            Embedding size.

        nb_filters_1 : int
            Filter size for each layer Conv1D.

        conv_width_1 : int
            Convolution 2D size

        ktop_max: int
            is the fixed pooling parameter for the
            topmost convolutional layer (Sect. 3.2 - https://arxiv.org/pdf/1404.2188.pdf)
            For instance, in a network with three convolutional layers and ktop = 3

        nb_filters_2 : int
            Filter size for each layer Conv1D.

        conv_width_2 : int
            Convolution 2D size

        nb_of_layers : int
            Units for dense layer.

        dropout_rate : float
            Dropout parameter.

        nb_classes : int
            Numbers of final categories.

    Attributes
    -----------
        embedding : tf.keras.layers.Embedding
            Embedding layer for input vocabulary.

        widening_1 : Widening custom Layer

        widening_2 : Widening custom Layer

        conv_1 : MyConv2D custom Layer

        conv_2 : MyConv2D custom Layer

        fold: Folding custom Layer

        pool_1 : tf.keras.layers.GlobalMaxPool1D
            Max pooling operation for 1D temporal data.

        pool_2 : tf.keras.layers.GlobalMaxPool1D
            Max pooling operation for 1D temporal data.

        dense : tf.keras.layers.Dense
            Regular densely-connected NN layer, concatenate 1D Convolutions.

        flatten : tf.keras.layers.Flatten

        dropout : tf.keras.layers.Dropout
            Applies Dropout to dense_1.

    Examples:
    ---------
        ```python

        VOCAB_SIZE = tokenizer.vocab_size #66125

        EMB_DIM = 60
        NB_FILTERS_1 = 6
        CONV_WIDTH_1 = 7
        KTOP_MAX = 4
        NB_FILTERS_2 = 14
        CONV_WIDTH_2 = 5
        FOLD_PATCH = 2
        NB_OF_LAYERS = 2
        DROPOUT_RATE = 0.1

        NB_CLASSES = 2

        EPOCHS=5

        Dcnn = DCNN(vocab_size=VOCAB_SIZE,
                    emb_dim=EMB_DIM,
                    nb_filters_1=NB_FILTERS_1,
                    conv_width_1=CONV_WIDTH_1,
                    ktop_max=KTOP_MAX,
                    nb_filters_2=NB_FILTERS_2,
                    conv_width_2=CONV_WIDTH_2,
                    fold_patch=FOLD_PATCH,
                    nb_of_layers=NB_OF_LAYERS,
                    dropout_rate=DROPOUT_RATE,
                    nb_classes=NB_CLASSES)

        if NB_CLASSES == 2:
            Dcnn.compile(loss="binary_crossentropy",
                        optimizer="adagrad",
                        metrics=["accuracy"])
        else:
            Dcnn.compile(loss="sparse_categorical_crossentropy",
                        optimizer="adagrad",
                        metrics=["sparse_categorical_accuracy"])

        Dcnn.fit(train_dataset,
                epochs=EPOCHS
                )

        results = Dcnn.evaluate(test_dataset)
        print(results)
        >>> [0.5345908999443054, 0.7479830384254456]
        ```
    """
    def __init__(self,
                 vocab_size,
                 emb_dim=48,
                 nb_filters_1=6,
                 conv_width_1=7,
                 ktop_max=4,
                 nb_filters_2=14,
                 conv_width_2=5,
                 fold_patch=2,
                 nb_of_layers=2,
                 padding_value=0,
                 dropout_rate=0.1,
                 nb_classes=2,
                 name="dcnn",
                 **kwargs):
        super(DCNN_Advanced, self).__init__(name=name, **kwargs)

        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.widening_1 = Widening(conv_width_1-1,
                                   axis=1)
        self.conv_1 = MyConv2D(nb_filters=nb_filters_1,
                               conv_width=conv_width_1,
                               padding="valid",
                               emb_dim=emb_dim)
        self.pool_1 = KMaxPooling(ktop_max,
                                  nb_of_layers,
                                  1)

        self.widening_2 = Widening(conv_width_2-1,
                                   axis=1)
        self.conv_2 = MyConv2D(nb_filters=nb_filters_2,
                               conv_width=conv_width_2,
                               padding="valid",
                               emb_dim=emb_dim)
        #self.fold = Folding(patch_size=fold_patch)
        self.fold = Folding()
        self.pool_2 = KMaxPooling(ktop_max)

        self.dropout = layers.Dropout(rate=dropout_rate)
        self.flatten = layers.Flatten()
        if nb_classes == 2:
            self.dense = layers.Dense(1, activation="sigmoid")
        else:
            self.dense = layers.Dense(nb_classes,
                                      activation="softmax")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = tf.expand_dims(x, axis=-1)

        x = self.widening_1(x)
        x = self.conv_1(x)
        x = self.pool_1(x)

        x = self.widening_2(x)
        x = self.conv_2(x)
        x = self.fold(x)
        x = self.pool_2(x)

        x = self.dropout(x, training)
        x = self.flatten(x)
        x = self.dense(x)
        return x
