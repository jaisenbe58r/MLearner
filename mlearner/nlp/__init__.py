"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from .cnn import DCNN
from .utils import Processor_data, open_txt
from mlearner.nlp import helpers

__all__ = ["DCNN", "Processor_data", "open_txt", "helpers"]

