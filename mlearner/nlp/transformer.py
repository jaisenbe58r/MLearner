"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""


import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import layers

from ..utils.keras import keras_checkpoint


"""
Based on:
https://www.udemy.com/course/procesamiento-del-lenguaje-natural/learn/lecture/21502260#overview

Author: Juan Gabriel Gomila
Course: "Procesamiento del Lenguaje Natural Moderno en Python.

Paper original: All you need is Attention https://arxiv.org/pdf/1706.03762.pdf
"""


class PositionalEncoding(layers.Layer):
    """
    Positional Encoding: Fórmula de la Codificación Posicional:
        $PE_{(pos,2i)} = \sin(pos/10000^{2i/dmodel})$
        $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/dmodel})$
    """
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):  # pos: (seq_length, 1) i: (1, d_model)
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles  # (seq_length, d_model)

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)


"""
Attention
"""


def scaled_dot_product_attention(queries, keys, values, mask):
    """
    Cálculo de la Atención:

        $Attention(Q, K, V ) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V $
    """
    product = tf.matmul(queries, keys, transpose_b=True)

    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9)

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)

    return attention


class MultiHeadAttention(layers.Layer):
    """
    Sub capa de atención de encabezado múltiple

    Parameters
    ----------
        nb_proj:
            Número de projecciones
    """
    def __init__(self, nb_proj):
        super(MultiHeadAttention, self).__init__()
        self.nb_proj = nb_proj

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0

        self.d_proj = self.d_model // self.nb_proj

        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size):  # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,
                 -1,
                 self.nb_proj,
                 self.d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape)  # (batch_size, seq_length, nb_proj, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3])  # (batch_size, nb_proj, seq_length, d_proj)

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_dot_product_attention(queries, keys, values, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention,
                                      shape=(batch_size, -1, self.d_model))

        outputs = self.final_lin(concat_attention)
        return outputs


class EncoderLayer(layers.Layer):
    """
    Capa de Codificación.

    Parameters
    ----------
        FFN_units:
            Número de capas densas.

        nb_proj:
            Número de projecciones.

        dropout_rate: Float
            Dropout parameter.
    """
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.FFN_units, activation="relu")
        self.dense_2 = layers.Dense(units=self.d_model)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs,
                                              inputs,
                                              inputs,
                                              mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)

        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        outputs = self.norm_2(outputs + attention)

        return outputs


class Encoder(layers.Layer):
    """
    Codificación.

    Parameters
    ----------
        nb_layers:
            Número de capas del transformer.

        FFN_units:
            Número de capas densas.

        nb_proj:
            Número de projecciones.

        dropout_rate: Float
            Dropout parameter.

        vocab_size:
            Vocabulary size of the algorithm input text.

        d_model: int
            Profundidad del embeding.
    """
    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [EncoderLayer(FFN_units,
                                        nb_proj,
                                        dropout_rate)
                           for _ in range(nb_layers)]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs


class DecoderLayer(layers.Layer):
    """
    Capa de Descodificación.

    Parameters
    ----------
        FFN_units:
            Número de capas densas.

        nb_proj:
            Número de projecciones.

        dropout_rate: Float
            Dropout parameter.
    """
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        # Self multi head attention
        self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        # Multi head attention combinado con la salida del encoder
        self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        # Feed foward
        self.dense_1 = layers.Dense(units=self.FFN_units,
                                    activation="relu")
        self.dense_2 = layers.Dense(units=self.d_model)
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_attention_1(inputs,
                                                inputs,
                                                inputs,
                                                mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention + inputs)

        attention_2 = self.multi_head_attention_2(attention,
                                                  enc_outputs,
                                                  enc_outputs,
                                                  mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)

        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)

        return outputs


class Decoder(layers.Layer):
    """
    Descodificación.

    Parameters
    ----------
        nb_layers:
            Número de capas del transformer.

        FFN_units:
            Número de capas densas.

        nb_proj:
            Número de projecciones.

        dropout_rate: Float
            Dropout parameter.

        vocab_size:
            Vocabulary size of the algorithm input text.

        d_model: int
            Profundidad del embeding.
    """
    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.nb_layers = nb_layers

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)

        self.dec_layers = [DecoderLayer(FFN_units,
                                        nb_proj,
                                        dropout_rate)
                           for _ in range(nb_layers)]

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs,
                                         enc_outputs,
                                         mask_1,
                                         mask_2,
                                         training)
        return outputs


class Transformer(tf.keras.Model):
    """
    Transformer

    Parameters
    ----------
        vocab_size_enc:
            Tamaño vocabulario entrada del transformer.

        vocab_size_dec:
            Tamaño vocabulario salida del transformer.

        nb_layers:
            Número de capas del transformer.

        FFN_units:
            Número de capas densas.

        nb_proj:
            Número de projecciones.

        dropout_rate: Float
            Dropout parameter.

        vocab_size:
            Vocabulary size of the algorithm input text.

        d_model: int
            Profundidad del embeding.

        name: str
            Nombre asociado a la instancia de la clase Transformer.

    Attributes
    -----------
        encoder : tf.keras.layers.Embedding
            encoder custom Layer

        decoder : tf.keras.layers.Embedding
            decoder custom Layer

    Examples:
    ---------
        ```python

        BATCH_SIZE = 64
        BUFFER_SIZE = 20000

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

        dataset = dataset.cache()
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        tf.keras.backend.clear_session()

        # Hiper Parámetros
        D_MODEL = 128 # 512
        NB_LAYERS = 4 # 6
        FFN_UNITS = 512 # 2048
        NB_PROJ = 8 # 8
        DROPOUT_RATE = 0.1 # 0.1

        model_Transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                                vocab_size_dec=VOCAB_SIZE_ES,
                                d_model=D_MODEL,
                                nb_layers=NB_LAYERS,
                                FFN_units=FFN_UNITS,
                                nb_proj=NB_PROJ,
                                dropout_rate=DROPOUT_RATE)

        Transformer_train(model_Transformer,
                        dataset,
                        d_model=D_MODEL,
                        train=TRAIN,
                        epochs=1,
                        checkpoint_path="ckpt/",
                        max_to_keep=5)

        # Evaluate
        def evaluate(inp_sentence):
            inp_sentence = \
                [VOCAB_SIZE_EN-2] + processor_en.tokenizer.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
            enc_input = tf.expand_dims(inp_sentence, axis=0)

            output = tf.expand_dims([VOCAB_SIZE_ES-2], axis=0)

            for _ in range(MAX_LENGTH):
                predictions = model_Transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)

                prediction = predictions[:, -1:, :]

                predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

                if predicted_id == VOCAB_SIZE_ES-1:
                    return tf.squeeze(output, axis=0)

                output = tf.concat([output, predicted_id], axis=-1)

            return tf.squeeze(output, axis=0)

        def translate(sentence):
            output = evaluate(sentence).numpy()

            predicted_sentence = processor_es.tokenizer.decode(
                [i for i in output if i < VOCAB_SIZE_ES-2]
            )

            print("Entrada: {}".format(sentence))
            print("Traducción predicha: {}".format(predicted_sentence))

        translate("This is a problem we have to solve.")

            >>> Entrada: This is a problem we have to solve.
            >>> Traducción predicha: Es un problema que debemos resolver.

        ```
    """
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)

        self.encoder = Encoder(nb_layers,
                               FFN_units,
                               nb_proj,
                               dropout_rate,
                               vocab_size_enc,
                               d_model)
        self.decoder = Decoder(nb_layers,
                               FFN_units,
                               nb_proj,
                               dropout_rate,
                               vocab_size_dec,
                               d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_ouput")

    def create_padding_mask(self, seq):  # seq: (batch_size, seq_length)
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )
        dec_mask_2 = self.create_padding_mask(enc_inputs)

        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)

        outputs = self.last_linear(dec_outputs)

        return outputs


class CustomSchedule_transformer(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom Learning Rate Schedule:
        (5.3.) https://arxiv.org/pdf/1706.03762.pdf

    Parameters
    ----------
        d_model: int
            Número de projecciones

        warmup_steps: int

    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule_transformer, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(target, pred):
    """
    Función de cáculo de pérdidas.

    Se calcula a partir de elementos no nulos, por ello
    extraemos elementos del padding.
    """
    # Mascara para deshacerme de los "ceros" del padding
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction="none")
    loss_ = loss_object(target, pred)

    # Equiparar el tipo de dato de la mascara al de la perdida
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Multiplicamos elemento a elemento.
    loss_ *= mask

    # Media por bloques (media de los resultados no nulos)
    return tf.reduce_mean(loss_)


def Transformer_train(Transformer,
                        dataset,
                        d_model,
                        epochs,
                        train=True,
                        beta_1=0.9,
                        beta_2=0.98,
                        epsilon=1e-9,
                        checkpoint_path="ckpt/",
                        max_to_keep=5):
    """
    Entrenamiento Transformer Customizado
    """
    # Custom Learning Rate Schedule
    leaning_rate = CustomSchedule_transformer(d_model)
    # Loss function
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    # Optimizador
    optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                            beta_1=beta_1,
                                            beta_2=beta_2,
                                            epsilon=epsilon)
    # Checkpoints
    ckpt_manager = keras_checkpoint(Transformer,
                                    optimizer,
                                    checkpoint_path=checkpoint_path,
                                    max_to_keep=5)
    # Grafo estatico
    @tf.function
    def train_step(enc_inputs, dec_inputs, dec_outputs_real):
        with tf.GradientTape() as tape:
            predictions = Transformer(enc_inputs, dec_inputs, True)
            loss = loss_function(dec_outputs_real, predictions)

        gradients = tape.gradient(loss, Transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Transformer.trainable_variables))

        train_accuracy(dec_outputs_real, predictions)
        return loss

    # Bucle de Entrenamiento
    if train:
        for epoch in range(epochs):
            print("Inicio del epoch {}".format(epoch+1))
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (enc_inputs, targets)) in enumerate(dataset):
                dec_inputs = targets[:, :-1]
                dec_outputs_real = targets[:, 1:]

                loss = train_step(enc_inputs, dec_inputs, dec_outputs_real)
                train_loss(loss)

                if batch % 50 == 0:
                    print("Epoch {} Lote {} Pérdida {:.4f} Precisión {:.4f}".format(
                        epoch+1, batch, train_loss.result(), train_accuracy.result()))

            ckpt_save_path = ckpt_manager.save()
            print("Guardando checkpoint para el epoch {} en {}".format(epoch+1,
                                                                        ckpt_save_path))
            print("Tiempo que ha tardado 1 epoch: {} segs\n".format(time.time() - start))
