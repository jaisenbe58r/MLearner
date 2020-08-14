"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.data import data_gamma

a = 5
n = 1000
out_mean = 5
inc = 0.1


def test_data_len():
    data = data_gamma(a, n)
    assert(len(data) == n)


def test_type_a():
    with pytest.raises(TypeError):
        data_gamma(str(a), n)


def test_type_n():
    with pytest.raises(TypeError):
        data_gamma(a, str(n))


def test_float_n():
    with pytest.raises(TypeError):
        data_gamma(a, float(n))


def test_float_n_minor():
    with pytest.raises(NameError):
        data_gamma(a, 0)


def test_result_mean():
    data = data_gamma(a, n)
    assert(np.mean(data) > np.median(data))
