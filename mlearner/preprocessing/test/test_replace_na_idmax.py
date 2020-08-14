
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import FillNaTransformer_idmax


data = pd.DataFrame({"a": [2, 2, 2, float('nan'), 4, 5], "b": ["OK", "OK", float('nan'), "OK", "OK", "NOK"]})
data_out = pd.DataFrame({"a": [2.0, 2.0, 2.0, 2.0, 4.0, 5.0], "b": ["OK", "OK", "OK", "OK", "OK", "NOK"]})
col = ["a"]
col_test = ["a", "b"]
col_no_include = ["a", "c"]


def test_idmax_fitting_error():
    ft = FillNaTransformer_idmax(columns=col)
    with pytest.raises(AttributeError):
        ft.transform(data)


def test_idmax_col_fail_type():
    with pytest.raises(NameError):
        FillNaTransformer_idmax(columns=np.array(col_test))


def test_idmax_col_object():
    ft = FillNaTransformer_idmax(columns=col_test)
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out)


def test_idmax_tranf_equal():
    ft = FillNaTransformer_idmax(columns=col)
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out)


def test_idmax_tranf_equal_null():
    ft = FillNaTransformer_idmax()
    ft.fit(data)
    assert_frame_equal(ft.transform(data), data_out)


def test_idmax_invalid_fit_type():
    ft = FillNaTransformer_idmax(columns=col)
    with pytest.raises(NameError):
        ft.fit(data.values)


def test_idmax_invalid_transf_type():
    ft = FillNaTransformer_idmax(columns=col)
    ft.fit(data)
    with pytest.raises(NameError):
        ft.transform(data.values)


def test_idmax_col_no_include():
    ft = FillNaTransformer_idmax(columns=col_no_include)
    with pytest.raises(NameError):
        ft.fit(data)
