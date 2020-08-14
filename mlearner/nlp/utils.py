"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""


import tensorflow as tf
import tensorflow_datasets as tfds

import re
from bs4 import BeautifulSoup


class Processor_data():
    """
    The DCNN class corresponds to the Neural Convolution Network algorithm
    for Natural Language Processing.

    Parameters
    ----------
        mame:
            Instance class name

    Attributes
    -----------
        clean: function
            Modulo limpieza de texto por medio de expresiones regulares


    Examples:
    ---------
        ```python
        cols = ["sentiment", "id", "date", "query", "user", "text"]
        data = pd.read_csv(
            TRAIN,
            header=None,
            names=cols,
            engine="python",
            encoding="latin1"
        )
        data.drop(["id", "date", "query", "user"],
                axis=1,
                inplace=True)
        nlptrans = Processor()
        data_process = nlptrans.process_text(data)
        ```
    """
    def __init__(self,
                 name="NLP"):
        self.name = name

    def clean(self, data):
        """
        Clean text.
        """
        return [self._clean_text(_text) for _text in data]

    def _clean_text(self, text):
        """
        Regular expressions applied to text.
        """
        text = BeautifulSoup(text, "lxml").get_text()
        # Eliminamos la @ y su mención
        text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
        # Eliminamos los links de las URLs
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        # Nos quedamos solamente con los caracteres
        text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
        # Eliminamos espacios en blanco adicionales
        text = re.sub(r" +", ' ', text)

        return text

    def _tokenizer(self, data, target_vocab_size=2**16):
        """
        Tokenizador:
        """
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            data, target_vocab_size=target_vocab_size
        )

    def encode_data(self, data, target_vocab_size=2**16):
        """
        Encoder all text
        """
        self._tokenizer(data=data, target_vocab_size=target_vocab_size)
        return [self.tokenizer.encode(sentence) for sentence in data]

    def padding(self, data, value=0, padding='post'):
        """
        El Pdding es una forma especial de enmascaramiento donde los pasos enmascarados
        se encuentran al comienzo o al comienzo de una secuencia. El padding proviene de
        la necesidad de codificar datos de secuencia en lotes contiguos: para que todas las
        secuencias en un lote se ajusten a una longitud estándar dada, es necesario rellenar
        o truncar algunas secuencias.
        """
        _MAX_LEN = max([len(sentence) for sentence in data])
        return tf.keras.preprocessing.sequence.pad_sequences(data,
                                                                    value=value,
                                                                    padding=padding,
                                                                    maxlen=_MAX_LEN)

    def process_text(self, data):
        """
        Procesador completo de texto:
         - Limpieza con expresiones regulares
         - Tokenizador
         - Padding
        """
        data = self.clean(data)
        data = self.encode_data(data)
        return self.padding(data)
