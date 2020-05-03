"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from .wine import wine_data
from ._dummy_dataset import data_uniform, data_normal, data_gamma, create_dataset


__all__ = ["wine_data", "data_normal", "data_uniform", "data_gamma", "create_dataset"]
