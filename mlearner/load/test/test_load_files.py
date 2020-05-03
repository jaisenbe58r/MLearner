
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pytest
from mlearner.load import DataLoad

filename = "mlearner/data/data/Boston.csv"
n_len = 506
filename_txt = "mlearner/data/data/breast-cancer-wisconsin-data.txt"
filename_img = "mlearner/load/file/image.jpg"
filename_null = ""
data_error = np.array([1, 2, 3])


def test_import_csv():
    """Test import dataset."""
    DataLoad.load_data(filename, sep=",")


def test_load_dataframe():
    data = DataLoad.load_data(filename, sep=",")
    DataLoad(data)


def test_import_txt():
    DataLoad.load_data(filename_txt)


def test_import_fail():
    with pytest.raises(UnicodeDecodeError):
        DataLoad.load_data(filename_img)


def test_import_null():
    with pytest.raises(FileNotFoundError):
        DataLoad.load_data(filename_null)


def test_load_dataframe_type_error():
    with pytest.raises(TypeError):
        DataLoad.load_dataframe(data_error)


def test_len_dataframe():
    data = DataLoad.load_data(filename, sep=",")
    print(len(data))
    assert(len(data) == n_len)
