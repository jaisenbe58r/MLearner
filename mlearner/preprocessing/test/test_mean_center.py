
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from mlearner.preprocessing import MeanCenterer


X1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
listX1 = X1.tolist()
pdX1 = pd.DataFrame(data=X1)

X1_out = np.array([[-1.5, -1.5, -1.5], [1.5,  1.5,  1.5]])
pdX1_out = pd.DataFrame(data=X1_out)

X2 = np.array([[1.0, 'N', 3.0], [4.0, 'Y', 6.0]])
pdX2 = pd.DataFrame(data=X2)


def test_fitting_error_numpy():
    mc = MeanCenterer()
    with pytest.raises(AttributeError):
        mc.transform(X1)


def test_array_mean_centering_numpy():
    mc = MeanCenterer()
    mc.fit(X1)
    np.testing.assert_allclose(mc.transform(X1), X1_out, rtol=1e-03)


def test_array_mean_centering_pandas():
    mc = MeanCenterer()
    mc.fit(pdX1)
    out = mc.transform(pdX1)
    np.testing.assert_allclose(out, pdX1_out, rtol=1e-03)


def test_columns_categorical_numpy():
    mc = MeanCenterer()
    with pytest.raises(NameError):
        mc.fit(X2)


def test_columns_categorical_pandas():
    mc = MeanCenterer()
    with pytest.raises(NameError):
        mc.fit(pdX2)


def test_invalid_type_fit():
    mc = MeanCenterer()
    with pytest.raises(NameError):
        mc.fit(listX1)


def test_invalid_type_transform():
    mc = MeanCenterer()
    mc.fit(X1)
    with pytest.raises(NameError):
        mc.transform(listX1)
