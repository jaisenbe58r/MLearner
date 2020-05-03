
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import ReplaceTransformer


data = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"], "b": [1, 1, 1, 1, 1, 0]})
data_out1 = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"], "b": ["OK", "OK", "OK", "OK", "OK", "NOK"]})
data_out2 = pd.DataFrame({"a": ["OK", "OK", "OK", "OK", "OK", "NOK"], "b": [1, 1, 1, 1, 1, 0]})

col1 = ["a"]
col2 = ["b"]
col_no_include = ["a", "c"]

mapping2 = {1: "OK", 0: "NOK"}
mapping1 = {"car": "OK", "house": "NOK"}
mapping_no_exist = {"car": "OK", "horse": "NOK"}


def test_init_columns_type_error():
    with pytest.raises(TypeError):
        ReplaceTransformer(columns=np.array(col1))


def test_init_mapping_type_error():
    with pytest.raises(TypeError):
        ReplaceTransformer(mapping=list(mapping1))


def test_init_mapping_unspecified():
    with pytest.raises(NameError):
        ReplaceTransformer()


def test_fit_data_type():
    rt = ReplaceTransformer(columns=col1, mapping=mapping1)
    with pytest.raises(TypeError):
        rt.fit(np.array(data))


def test_fit_data_no_columns_exist():
    rt = ReplaceTransformer(columns=col_no_include, mapping=mapping1)
    with pytest.raises(NameError):
        rt.fit(data)


def test_fit_data_no_keys_exist():
    rt = ReplaceTransformer(columns=col1, mapping=mapping_no_exist)
    with pytest.raises(NameError):
        rt.fit(data)


def test_fitting_error():
    rt = ReplaceTransformer(columns=col1, mapping=mapping1)
    with pytest.raises(AttributeError):
        rt.transform(data)


def test_transform_data_type():
    rt = ReplaceTransformer(columns=col1, mapping=mapping1)
    rt.fit(data)
    with pytest.raises(TypeError):
        rt.transform(np.array(data))


def test_run_1():
    rt = ReplaceTransformer(columns=col1, mapping=mapping1)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out2)


def test_run_2():
    rt = ReplaceTransformer(columns=col2, mapping=mapping2)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out1)
