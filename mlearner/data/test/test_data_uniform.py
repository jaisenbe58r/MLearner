"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.data import data_uniform

a = 0
b = 1
n = 1000
out = 0.5
inc = 0.1


def test_data_len():
    data = data_uniform(a, b, n)
    assert(len(data) == n)


def test_type_a():
    with pytest.raises(TypeError):
        data_uniform(str(a), b, n)


def test_type_b():
    with pytest.raises(TypeError):
        data_uniform(a, str(b), n)


def test_type_n():
    with pytest.raises(TypeError):
        data_uniform(a, b, str(n))


def test_float_n():
    with pytest.raises(TypeError):
        data_uniform(a, b, float(n))


def test_float_n_minor():
    with pytest.raises(NameError):
        data_uniform(a, b, 0)


def test_result():
    data = data_uniform(a, b, n)
    np.testing.assert_allclose(np.mean(data), out, rtol=inc)
