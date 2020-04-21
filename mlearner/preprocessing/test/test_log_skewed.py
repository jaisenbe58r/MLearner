
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.preprocessing import FixSkewness
from mlearner.data import data_gamma, create_dataset

n = 1000
config = {
        'A': data_gamma(a=2.5, n=n),
        'B': data_gamma(a=3.5, n=n),
        'C': data_gamma(a=4, n=n),
        'D': data_gamma(a=5, n=n),
        'E': np.repeat("KO", n)
    }
data = create_dataset(config, n)
data_error = [1, 2, 3, 4, 1, 2, 3, 4]
col = ["D"]
col_error = np.array([1, 2])
col_object = ["E"]


def test_error_type_col():
    with pytest.raises(TypeError):
        FixSkewness(columns=col_error)


def test_error_type_dataframe():
    fs = FixSkewness(columns=col)
    with pytest.raises(NameError):
        fs.fit(data_error)


def test_error_type_dataframe_object_column():
    fs = FixSkewness(columns=col_object)
    with pytest.raises(NameError):
        fs.fit(data)


def test_fitting_error():
    """FixSkewness has not been fitted, yet."""
    fs = FixSkewness(columns=col)
    with pytest.raises(AttributeError):
        fs.transform(data)


def test_error_type_dataframe_transf():
    fs = FixSkewness(columns=col)
    fs.fit(data)
    with pytest.raises(NameError):
        fs.transform(data_error)
