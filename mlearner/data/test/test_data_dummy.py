"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.data import data_uniform, data_normal, data_gamma, create_dataset


n = 1000
config = {
        'A': data_uniform(0, 1, n),
        'B': data_normal(n),
        'C': data_normal(mu=5, sd=2, n=n),
        'D': data_gamma(a=5, n=n)
    }
config_bad = np.array([1, 2, 3])


def test_config_bad():
    with pytest.raises(TypeError):
        create_dataset(config_bad, n)


def test_type_n():
    with pytest.raises(TypeError):
        create_dataset(config, float(n))


def test_minor_n():
    with pytest.raises(NameError):
        create_dataset(config, 0)
