
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import ReplaceMulticlass


data = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"], "b": ["OK", "OK", "OK", "OK", "OK", "NOK"]})
data_out = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"], "b": [0, 0, 0, 0, 0, 1]})
data_out1 = pd.DataFrame({"a": [0, 0, 0, 0, 0, 1], "b": [0, 0, 0, 0, 0, 1]})


col = ["b"]
col1 = ["a", "b"]
col_no_include = ["a", "c"]


def test_init_columns_type_error():
    with pytest.raises(TypeError):
        ReplaceMulticlass(columns=np.array(col))


def test_fit_data_type():
    rt = ReplaceMulticlass(columns=col)
    with pytest.raises(TypeError):
        rt.fit(np.array(data))


def test_fit_data_no_columns_exist():
    rt = ReplaceMulticlass(columns=col_no_include)
    with pytest.raises(NameError):
        rt.fit(data)


def test_fitting_error():
    rt = ReplaceMulticlass(columns=col)
    with pytest.raises(AttributeError):
        rt.transform(data)


def test_transform_data_type():
    rt = ReplaceMulticlass(columns=col)
    rt.fit(data)
    with pytest.raises(TypeError):
        rt.transform(np.array(data))


def test_run_1():
    rt = ReplaceMulticlass(columns=col)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out)


def test_run_2():
    rt = ReplaceMulticlass(columns=col1)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out1)


def test_run_3():
    rt = ReplaceMulticlass()
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out1)
