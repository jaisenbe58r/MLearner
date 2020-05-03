"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.data import data_normal

mu = 10
sd = 2
n = 1000
out = 10
inc = 0.1


def test_data_len():
    data = data_normal(mu, sd, n)
    assert(len(data) == n)


def test_type_a():
    with pytest.raises(TypeError):
        data_normal(str(mu), sd, n)


def test_type_b():
    with pytest.raises(TypeError):
        data_normal(mu, str(sd), n)


def test_type_n():
    with pytest.raises(TypeError):
        data_normal(mu, sd, str(n))


def test_float_n():
    with pytest.raises(TypeError):
        data_normal(mu, sd, float(n))


def test_float_n_minor():
    with pytest.raises(NameError):
        data_normal(mu, sd, 0)


def test_result():
    data = data_normal(mu, sd, n)
    np.testing.assert_allclose(np.mean(data), out, rtol=inc)
