# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

import numpy as np
from mlearner.evaluate import bootstrap
from mlearner.utils import assert_raises


def test_defaults():
    rng = np.random.RandomState(123)
    x = rng.normal(loc=5., size=100)
    original, std_err, ci_bounds = bootstrap(x, func=np.mean, seed=123)
    assert round(original, 2) == 5.03
    assert round(std_err, 2) == 0.11
    assert round(ci_bounds[0], 2) == 4.80
    assert round(ci_bounds[1], 2) == 5.26


def test_fail_ci():
    msg = 'ci must be in range (0, 1)'
    assert_raises(AttributeError,
                  msg,
                  bootstrap,
                  np.array([1, 2, 3]),
                  np.mean,
                  1000,
                  95)


def test_fail_func():

    def f(x):
        return np.array([0, 1])

    msg = 'ci must be in range (0, 1)'
    assert_raises(AttributeError,
                  msg,
                  bootstrap,
                  np.array([1, 2, 3]),
                  f,
                  1000,
                  95)
