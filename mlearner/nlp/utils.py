"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import os
import pandas as pd
import numpy as np

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
    def __init__(
            self,
            target_vocab_size=2**16,
            language="en",
            value=0,
            padding='post',
            function=None,
            name="NLP"
            ):
        self.name = name
        self.target_vocab_size = target_vocab_size
        self.language = language
        self.value = value
        self.padding = padding
        self.function = function

    def clean(self, data):
        """
        Clean text.
        """
        if isinstance(data, np.ndarray):
            return [self._clean_text(_text[0]) for _text in data]

        elif isinstance(data, pd.core.frame.DataFrame) or isinstance(data, pd.core.series.Series):
            if data.isna().sum()[0] > 0:
                data = data.fillna(" ")
            return [self._clean_text(_text[0]) for _text in data.values]

        elif isinstance(data, list):
            return [self._clean_text(_text) for _text in data]

        else:
            data = data.split("\n")
            return [self._clean_text(_text) for _text in data]

    def apply_non_breaking_prefix(self, text, language="en"):
        """
        clean words with a period at the end to make it easier for us to use.

        Parameters
        ----------
            text:
                Text to apply cleaning.
            language: str
                Language a nonbreaking_prefix. options: en / es / fr.
        """
        this_dir, _ = os.path.split(__file__)
        _DATA_PATH = os.path.join(this_dir, "data", "Non-Breaking-Prefix")
        if language == "en":
            non_breaking_prefix = open_txt(os.path.join(_DATA_PATH, "P85-Non-Breaking-Prefix.en"))
        elif language == "es":
            non_breaking_prefix = open_txt(os.path.join(_DATA_PATH, "P85-Non-Breaking-Prefix.es"))
        elif language == "fr":
            non_breaking_prefix = open_txt(os.path.join(_DATA_PATH, "P85-Non-Breaking-Prefix.fr"))
        else:
            raise NameError(f"Language {language} not implemented")
        # Añadimos punto detras del prefijo
        non_breaking_prefix = non_breaking_prefix.split("\n")
        non_breaking_prefix = [' ' + pref + '.' for pref in non_breaking_prefix]
        for prefix in non_breaking_prefix:
            text = text.replace(prefix, prefix + '$$$')
        text = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", text)
        text = re.sub(r"\.\$\$\$", '', text)
        return text

    def _clean_text(self, text, split=None, non_breaking_prefix=True):
        """
        Regular expressions applied to text.

        ```python
        def Function_clean(non_breaking_prefix_en):
            text = BeautifulSoup(text, "lxml").get_text()
            # Eliminamos la @ y su mención
            text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
            # Eliminamos los links de las URLs
            text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
            # Nos quedamos solamente con los caracteres
            text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
            return text
        ````
        """
        if str(text) == 'nan':
            text = " "
            print(text)

        def Function_clean(text):
            """
            Función por defecto
            """
            text = BeautifulSoup(text, "lxml").get_text()
            # Eliminamos la @ y su mención
            text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
            # Eliminamos los links de las URLs
            text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
            # Nos quedamos solamente con los caracteres
            text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
            return text

        if self.function is None:
            text = Function_clean(text)
        else:
            text = self.function(text)

        if non_breaking_prefix:
            text = self.apply_non_breaking_prefix(text)
        # Eliminamos espacios en blanco adicionales
        text = re.sub(r" +", ' ', text)
        # Se aplica split. >> p.e. text.split("\n")
        if split is not None:
            text = text.split(split)
        return text

    def _tokenizer(self, data):
        """
        Tokenizador:
        """
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            data, target_vocab_size=self.target_vocab_size
        )

    def encode_data(self, data, eval=False):
        """
        Encoder all text
        """
        if not eval:
            self._tokenizer(data=data)
        return [self.tokenizer.encode(sentence) for sentence in data]

    def apply_padding(self, data, eval=False):
        """
        El Pdding es una forma especial de enmascaramiento donde los pasos enmascarados
        se encuentran al comienzo o al comienzo de una secuencia. El padding proviene de
        la necesidad de codificar datos de secuencia en lotes contiguos: para que todas las
        secuencias en un lote se ajusten a una longitud estándar dada, es necesario rellenar
        o truncar algunas secuencias.
        """
        if not eval:
            self._MAX_LEN = max([len(sentence) for sentence in data])
        return tf.keras.preprocessing.sequence.pad_sequences(
            data,
            value=self.value,
            padding=self.padding,
            maxlen=self._MAX_LEN
            )

    def process_text(self, data, eval=False, isclean=False, padding=True):
        """
        Procesador completo de texto:
         - Limpieza con expresiones regulares
         - Tokenizador
         - Padding
        """
        # Cleaning
        if not isclean:
            data = self.clean(data)
        # List of strings
        else:
            if isinstance(data, np.ndarray):
                data = [_text[0] for _text in data]
            elif isinstance(data, pd.core.frame.DataFrame) or isinstance(data, pd.core.series.Series):
                if data.isna().sum()[0] > 0:
                    data = data.fillna(" ")
                data = [_text[0] for _text in data.values]
            elif isinstance(data, list):
                data = [_text for _text in data]
            else:
                data = data.split("\n")
                data = [_text for _text in data]
        # Encoding
        data = self.encode_data(data, eval=eval)
        # Padding
        if padding:
            return self.apply_padding(data, eval)
        else:
            return data


def delete_max_length(data_in, data_out=None, max_length=20):
    """
    Delete phrases with more than `max_length` characters.
    """
    idx_to_remove = [count for count, sent in enumerate(data_in)
                        if len(sent) > max_length]
    print(idx_to_remove)
    for idx in reversed(idx_to_remove):
        del data_in[idx]
        if data_out is not None:
            del data_out[idx]

    if data_out is not None:
        idx_to_remove = [count for count, sent in enumerate(data_out)
                            if len(sent) > max_length]
        for idx in reversed(idx_to_remove):
            del data_in[idx]
            del data_out[idx]
        return data_in, data_out
    else:
        return data_in


def open_txt(filename, encoding="utf-8"):
    """
    Function to open a .txt and return list of phrases.

    Parameters
    ----------
        filename:
            Path where the file is hosted.

        encoding:
            Unicode and text encodings.
    """
    with open(filename, mode="r", encoding=encoding) as f:
        text = f.read()
    return text
