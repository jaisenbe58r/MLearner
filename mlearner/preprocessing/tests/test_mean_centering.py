# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT


import numpy as np
import pytest
from mlearner.preprocessing import MeanCenterer


def test_fitting_error():
    X1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mc = MeanCenterer()
    with pytest.raises(AttributeError):
        mc.transform(X1)


def test_array_mean_centering():
    X1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X1_out = np.array([[-1.0, -1.0, -1.0], [1.0,  1.0,  1.0]])
    mc = MeanCenterer()
    assert(mc.fit_transform(X1).all() == X1_out.all())


def test_list_mean_centering():
    X2 = [1.0, 2.0, 3.0]
    X2_out = np.array([-1.0, 0.0, 1.0])
    mc = MeanCenterer()
    assert(mc.fit_transform(X2).all().all() == X2_out.all())
