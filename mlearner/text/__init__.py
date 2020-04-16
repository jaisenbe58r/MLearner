# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

import sys

if sys.version_info >= (3, 0):
    from .names import generalize_names
    from .names import generalize_names_duplcheck

from .tokenizer import tokenizer_words_and_emoticons
from .tokenizer import tokenizer_emoticons

__all__ = ["generalize_names", "generalize_names_duplcheck",
           "tokenizer_words_and_emoticons", "tokenizer_emoticons"]
