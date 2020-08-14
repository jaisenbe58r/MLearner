
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from mlearner.preprocessing import FeatureDropper


data = pd.DataFrame({"a": [0, 1], "b": [10, 11], "c": [20, 21]})
data_transf = pd.DataFrame({"a": [0, 1], "c": [20, 21]})
col_drop = ["b"]


def test_fitting_error():
    """FeatureDropper has not been fitted, yet."""
    fd = FeatureDropper(drop=col_drop)
    with pytest.raises(AttributeError):
        fd.transform(data)


def test_coldrop_fail_type():
    with pytest.raises(NameError):
        FeatureDropper(drop=np.array(col_drop))


def test_tranf_equal():
    fd = FeatureDropper(drop=col_drop)
    fd.fit(data)
    np.testing.assert_allclose(fd.transform(data), data_transf, rtol=1e-03)


def test_tranf_equal_null():
    fd = FeatureDropper()
    fd.fit(data)
    np.testing.assert_allclose(fd.transform(data), data, rtol=1e-03)


def test_invalid_fit_type():
    fd = FeatureDropper(drop=col_drop)
    with pytest.raises(NameError):
        fd.fit(data.values)


def test_invalid_transf_type():
    fd = FeatureDropper(drop=col_drop)
    fd.fit(data)
    with pytest.raises(NameError):
        fd.transform(data.values)
