
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import OneHotEncoder

config = {
        'A': ["Y", "N", "Y", "N"],
        'B': [1, 2, 1, 2]
    }
data = pd.DataFrame(config)
config_result = {
        'B': [1, 2, 1, 2],
        'A_Y': [np.uint8(1), np.uint8(0), np.uint8(1), np.uint8(0)],
        'A_N': [np.uint8(0), np.uint8(1), np.uint8(0), np.uint8(1)]
    }
data_result = pd.DataFrame(config_result).astype(np.uint8)
config_result_1 = {
        'A_Y': [1, 0, 1, 0],
        'A_N': [0, 1, 0, 1],
        'B_1': [1, 0, 1, 0],
        'B_2': [0, 1, 0, 1]
    }
data_result_1 = pd.DataFrame(config_result_1).astype(np.uint8)
data_error = [1, 2, 3, 4, 1, 2, 3, 4]
col = ["A", "B"]
col_1 = ["A"]
numerical = ["B"]
col_error = np.array([1, 2])
numerical_error = np.array([1, 2])


def test_error_type_col():
    with pytest.raises(TypeError):
        OneHotEncoder(columns=col_error)


def test_error_type_numerical():
    with pytest.raises(TypeError):
        OneHotEncoder(numerical=numerical_error)


def test_error_type_dataframe():
    fs = OneHotEncoder(columns=col)
    with pytest.raises(NameError):
        fs.fit(data_error)


def test_fitting_error():
    """OneHotEncoder has not been fitted, yet."""
    fs = OneHotEncoder(columns=col)
    with pytest.raises(AttributeError):
        fs.transform(data)


def test_fitting_error_col_None():
    """OneHotEncoder has not been fitted, yet."""
    fs = OneHotEncoder()
    with pytest.raises(AttributeError):
        fs.transform(data)


def test_error_type_dataframe_transf():
    fs = OneHotEncoder(columns=col)
    fs.fit(data)
    with pytest.raises(NameError):
        fs.transform(data_error)


def test_eval_result():
    fs = OneHotEncoder(columns=col)
    fs.fit(data)
    assert_frame_equal(fs.transform(data), data_result_1)


def test_eval_result0():
    fs = OneHotEncoder()
    fs.fit(data)
    assert_frame_equal(fs.transform(data).astype(np.uint8), data_result)


def test_eval_result_1():
    fs = OneHotEncoder(columns=col_1, numerical=numerical)
    fs.fit(data)
    assert_frame_equal(fs.transform(data), data_result_1)
