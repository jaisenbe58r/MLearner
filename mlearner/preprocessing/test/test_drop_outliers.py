
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
from pandas.testing import assert_frame_equal
from unittest import mock
from mlearner.preprocessing import DropOutliers
from mlearner.load import DataLoad

matplotlib.use('Template')

data = pd.DataFrame({"a": [0, 1, 1, 0, 1, 1], "b": [10, 11, 12, 13, 11, 100], "c": ["OK", "OK", "NOK", "OK", "OK", "NOK"]})
data_transf = pd.DataFrame({"a": [0, 1, 1, 0, 1], "b": [10, 11, 12, 13, 11], "c": ["OK", "OK", "NOK", "OK", "OK"]})
col = ["b"]
col_object = ["b", "c"]


def test_display_fail_type():
    with pytest.raises(TypeError):
        DropOutliers(display="a")


def test_coldrop_fail_type():
    with pytest.raises(TypeError):
        DropOutliers(features=np.array(col))


def test_coldrop_fail_fit_data_type():
    fdo = DropOutliers(col_object)
    with pytest.raises(TypeError):
        fdo.fit(np.asarray(data))


def test_features_col_object():
    fdo = DropOutliers(col_object)
    with pytest.raises(ValueError):
        fdo.fit(data)


def test_coldrop_fail_tranf_data_type():
    fdo = DropOutliers(col)
    fdo.fit(data)
    with pytest.raises(TypeError):
        fdo.transform(np.asarray(data))


def test_fitting_error():
    """DropOutliers has not been fitted, yet."""
    fd = DropOutliers(features=col)
    with pytest.raises(AttributeError):
        fd.transform(data)


def test_tranf_equal():
    fd = DropOutliers(features=col)
    fd.fit(data)
    assert_frame_equal(fd.transform(data), data_transf)


def test_tranf_equal_null():
    fd = DropOutliers()
    fd.fit(data)
    assert_frame_equal(fd.transform(data), data_transf)


def test_invalid_fit_type():
    fd = DropOutliers(features=col)
    with pytest.raises(TypeError):
        fd.fit(data.values)


def test_invalid_fit_type1():
    fd = DropOutliers(features=col)
    fd.fit(data)
    with pytest.raises(TypeError):
        fd.transform(data.values)


def test_tranf_equal_null_display():
    fd = DropOutliers(display=True)
    fd.fit(data)
    assert_frame_equal(fd.transform(data), data_transf)


def test_tranf_equal_null_display1():
    fd = DropOutliers(features=col, display=True)
    fd.fit(data)
    assert_frame_equal(fd.transform(data), data_transf)


def test_all():
    filename = "mlearner/data/data/titanic3.csv"
    dataset = DataLoad.load_data(filename, sep=",")
    fd = DropOutliers(display=True)
    fd.fit(dataset.data)
    fd.transform(dataset.data)
