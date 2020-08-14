
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import ExtractCategories


data = pd.DataFrame({"a": ["house", "car", "car", "glass", "house", "car"], "b": [0, 1, 1, 2, 0, 1]})
data_out02 = pd.DataFrame({"a": ["house", "house", "glass"], "b": [0, 0, 2]})
data_out01 = pd.DataFrame({"a": ["house", "house", "car", "car", "car"], "b": [0, 0, 1, 1, 1]})

cat02 = [0, 2]
cat01 = [0, 1]
cat_no_include = [3]

target = ["b"]
target_null = []
target_error = ["c"]
target_max = ["a", "b"]


def test_init_categories_type_error():
    with pytest.raises(TypeError):
        ExtractCategories(categories=np.array(cat02))


def test_init_categories_empty_error():
    with pytest.raises(NameError):
        ExtractCategories(categories=[])


def test_init_target_type_error():
    with pytest.raises(TypeError):
        ExtractCategories(target=np.array(target))


def test_init_target_empty_error():
    with pytest.raises(NameError):
        ExtractCategories(target=[])


def test_init_target_max_error():
    with pytest.raises(NameError):
        ExtractCategories(target=target_max)


def test_fit_data_type():
    rt = ExtractCategories()
    with pytest.raises(TypeError):
        rt.fit(np.array(data))


def test_fit_data_no_target_exist():
    rt = ExtractCategories(categories=cat01, target=target_error)
    with pytest.raises(NameError):
        rt.fit(data)


def test_fit_data_no_categories_exist():
    rt = ExtractCategories(categories=cat_no_include, target=target)
    with pytest.raises(NameError):
        rt.fit(data)


def test_no_fit():
    rt = ExtractCategories(categories=cat01, target=target)
    with pytest.raises(AttributeError):
        rt.transform(data)


def test_transform_data_type():
    rt = ExtractCategories(categories=cat01, target=target)
    rt.fit(data)
    with pytest.raises(TypeError):
        rt.transform(np.array(data))


def test_run_1():
    rt = ExtractCategories(categories=cat01, target=target)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out01)


def test_run_2():
    rt = ExtractCategories(categories=cat02, target=target)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out02)


def test_run_3():
    rt = ExtractCategories(categories=cat02)
    rt.fit(data)
    assert_frame_equal(rt.transform(data), data_out02)
