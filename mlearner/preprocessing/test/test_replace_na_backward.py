
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import FillNaTransformer_backward


data = pd.DataFrame({"a": [2, 2, 2, float('nan'), 4, 5], "b": ["OK", "OK", "OK", float('nan'), "NOK", "OK"]})
data_out_a = pd.DataFrame({"a": [2.0, 2.0, 2.0, 4.0, 4.0, 5.0], "b": ["OK", "OK", "OK", float('nan'), "NOK", "OK"]})
data_out_b = pd.DataFrame({"a": [2, 2, 2, float('nan'), 4, 5], "b": ["OK", "OK", "OK", "NOK", "NOK", "OK"]})
data_out = pd.DataFrame({"a": [2.0, 2.0, 2.0, 4.0, 4.0, 5.0], "b": ["OK", "OK", "OK", "NOK", "NOK", "OK"]})
col_a = ["a"]
col_b = ["b"]
col_test = ["a", "b"]
col_no_include = ["a", "c"]


def test_fitting_error():
    """FeatureDropper has not been fitted, yet."""
    ft = FillNaTransformer_backward(columns=col_a)
    with pytest.raises(AttributeError):
        ft.transform(data)


def test_col_fail_type():
    with pytest.raises(NameError):
        FillNaTransformer_backward(columns=np.array(col_test))


def test_tranf_equal_a():
    ft = FillNaTransformer_backward(columns=col_a)
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out_a)


def test_tranf_equal_b():
    ft = FillNaTransformer_backward(columns=col_b)
    ft.fit(data)
    print(ft.transform(data))
    assert_frame_equal(ft.transform(data), data_out_b)


def test_tranf_equal_all():
    ft = FillNaTransformer_backward(columns=col_test)
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out)


def test_tranf_equal_null():
    ft = FillNaTransformer_backward()
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out)


def test_invalid_fit_type():
    ft = FillNaTransformer_backward(columns=col_a)
    with pytest.raises(NameError):
        ft.fit(data.values)


def test_invalid_transf_type():
    ft = FillNaTransformer_backward(columns=col_a)
    ft.fit(data)
    with pytest.raises(NameError):
        ft.transform(data.values)


def test_col_no_include():
    ft = FillNaTransformer_backward(columns=col_no_include)
    with pytest.raises(NameError):
        ft.fit(data)
