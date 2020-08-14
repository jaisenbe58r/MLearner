
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import pytest
from mlearner.preprocessing import DataExploratory


data = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"],
                        "b": ["car1", "car1", float('nan'), "car1", "car1", "house1"],
                        "t": [1, 1, 1, 1, 1, 0]})
data_no_null = pd.DataFrame({"a": ["car", "car", "car", "car", "car", "house"],
                                "t": [1, 1, 1, 1, 1, 0]})
data_columns = ['a', 'b', 't']
cat_columns = ['a', 'b']
num_columns = ['t']


def test_init_type_data():
    with pytest.raises(TypeError):
        DataExploratory(np.array(data))


def test_dtypes():
    dc = DataExploratory(data)
    dc.dtypes()


def test_missing_values():
    dc = DataExploratory(data)
    dc.missing_values()


def test_isNull_values():
    dc = DataExploratory(data)
    dc.isNull()


def test_isNull_NoNull_values():
    dc = DataExploratory(data_no_null)
    dc.isNull()


def test_view_features_values():
    dc = DataExploratory(data)
    columns = dc.view_features()
    assert(columns == data_columns)


def test_categorical_vs_numerical_values():
    dc = DataExploratory(data)
    categorical_list, numerical_list = dc.categorical_vs_numerical()
    assert(categorical_list == cat_columns)
    assert(numerical_list == num_columns)


def test_type_object():
    dc = DataExploratory(data)
    var = dc.type_object()
    assert(var == cat_columns)


def test_not_type_object():
    dc = DataExploratory(data)
    var = dc.not_type_object()
    assert(var == num_columns)
