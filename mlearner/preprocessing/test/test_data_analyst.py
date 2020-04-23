
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import pytest
from mlearner.preprocessing import DataAnalyst
import matplotlib

matplotlib.use('Template')

data = pd.DataFrame({"a": [0., 1., 1., 0., 1., 1.], "b": [10, 11, 12, 13, 11, 100], "c": ["OK", "OK", "NOK", "OK", "OK", "NOK"]})
col = ["b"]


def test_init_type_data():
    with pytest.raises(TypeError):
        DataAnalyst(np.array(data))


def test_boxplot_features_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.boxplot(features=np.array(col))


def test_boxplot_features_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=[])


def test_boxplot_features_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=["d"])


def test_boxplot_features_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=["c"], target=["a"])


def test_boxplot_target_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.boxplot(features=["a"], target=np.array(col))


def test_boxplot_target_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["d"])


def test_boxplot_target_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["a", "b"])


def test_boxplot_target_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col)


def test_boxplot_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["c"], save_image=True, path="")


def test_boxplot_save_image():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=["a", "b"], target=["c"], display=True)
