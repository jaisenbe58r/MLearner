
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats


def data_uniform(a, b, n):
    """Generate a Uniform data distribution.

    Attributes
    ----------
    a : `int` or `float`
        manimum value of the entire dataset.
    b : `int` or `float`
        maximum value of the entire dataset.
    n : `int`
        number of data in the dataset.

    Returns
    -------
    data : Uniform data distribution.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/data_uniform/


    """
    if not isinstance(a, int) and not isinstance(a, float):
        raise TypeError("Invalid type for a : {}".format(type(a)))
    if not isinstance(b, int) and not isinstance(b, float):
        raise TypeError("Invalid type for a : {}".format(type(b)))
    if not isinstance(n, int):
        raise TypeError("Invalid type for a : {}".format(type(n)))

    if n <= 0:
        raise NameError("'n' must be an integer greater than zero")

    data = np.random.uniform(a, b, n)
    return data


def data_normal(mu=0, sd=1, n=100):
    """Generate a Normal data distribution.

    Attributes
    ----------
    mu : `int` or `float`
        mean value.
    sd : `int` or `float`
        standard deviation.
    n : `int`
        number of data in the dataset.

    Returns
    -------
    data : Uniform data distribution.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/data_normal/


    """
    if not isinstance(mu, int) and not isinstance(mu, float):
        raise TypeError("Invalid type for a : {}".format(type(mu)))
    if not isinstance(sd, int) and not isinstance(sd, float):
        raise TypeError("Invalid type for a : {}".format(type(sd)))
    if not isinstance(n, int):
        raise TypeError("Invalid type for a : {}".format(type(n)))

    if n <= 0:
        raise NameError("'n' must be an integer greater than zero")

    data = mu + sd * np.random.randn(n)
    return data


def data_gamma(a=5, n=100):
    """Generate a Gamma data distribution.

    Attributes
    ----------
    a : `int` or `float`
        Parameter form.
    n : `int`
        number of data in the dataset.

    Returns
    -------
    data : Uniform data distribution.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/data_gamma/


    """
    if not isinstance(a, int) and not isinstance(a, float):
        raise TypeError("Invalid type for a : {}".format(type(a)))
    if not isinstance(n, int):
        raise TypeError("Invalid type for a : {}".format(type(n)))

    if n <= 0:
        raise NameError("'n' must be an integer greater than zero")

    gamma = stats.gamma(a)
    data = gamma.rvs(n)
    return data


def create_dataset(config, n):
    """Generate a Dataset.

    Attributes
    ----------
    config : `dict`
        Dictionary for dataset configuration:
            p.e.:
            ```
            dict = {
                'A' : data_uniform(0, 1, n),
                'B' : data_normal(n),
                'C' : data_normal(mu=5, sd=2, n=n),
                'D' : data_gamma(a=5, n=n)
            }
            ```
    n : `int`
        number of data in the dataset.

    Returns
    -------
    data : Dataset

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/create_dataset/

    """
    if not isinstance(config, dict):
        raise TypeError("Invalid type for 'config' : {}".format(type(config)))

    if not isinstance(n, int):
        raise TypeError("Invalid type for a : {}".format(type(n)))

    if n <= 0:
        raise NameError("'n' must be an integer greater than zero")

    data = pd.DataFrame(config)
    return data
