
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import FillNaTransformer_value


data = pd.DataFrame({"a": [2, 2, 2, float('nan'), 4, 5], "b": ["OK", 2, 2, 2, 4, 5]})
data_out = pd.DataFrame({"a": [2.0, 2.0, 2.0, "Empty", 4.0, 5.0], "b": ["OK", 2, 2, 2, 4, 5]})
value = "Empty"
value_NOK = np.array([1, 2, 3])
col = ["a"]
col_test = ["a", "b"]
col_no_include = ["a", "c"]


def test_fitting_error():
    """FeatureDropper has not been fitted, yet."""
    ft = FillNaTransformer_value(columns=col)
    with pytest.raises(AttributeError):
        ft.transform(data)


def test_col_fail_type():
    with pytest.raises(NameError):
        FillNaTransformer_value(columns=np.array(col_test))


def test_col_object():
    ft = FillNaTransformer_value(columns=col)
    ft.fit(data, value=value)
    assert_frame_equal(ft.transform(data), data_out)


def test_value_none():
    ft = FillNaTransformer_value(columns=col_test)
    with pytest.raises(NameError):
        ft.fit(data)


def test_value_incorrect():
    ft = FillNaTransformer_value(columns=col_test)
    with pytest.raises(NameError):
        ft.fit(data, value=value_NOK)


def test_tranf_equal():
    ft = FillNaTransformer_value(columns=col)
    ft.fit(data, value=value)
    assert_frame_equal(ft.transform(data), data_out)


def test_tranf_equal_null():
    ft = FillNaTransformer_value()
    ft.fit(data, value=value)
    assert_frame_equal(ft.transform(data), data_out)


def test_invalid_fit_type():
    ft = FillNaTransformer_value(columns=col)
    with pytest.raises(NameError):
        ft.fit(data.values)


def test_invalid_transf_type():
    ft = FillNaTransformer_value(columns=col)
    ft.fit(data, value=value)
    with pytest.raises(NameError):
        ft.transform(data.values)


def test_col_no_include():
    ft = FillNaTransformer_value(columns=col_no_include)
    with pytest.raises(NameError):
        ft.fit(data, value=value)
