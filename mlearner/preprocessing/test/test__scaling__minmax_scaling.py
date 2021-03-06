
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import pytest
from mlearner.preprocessing import minmax_scaling


def test_minmax_scaling_arrayerror():
    """Test minmax type."""
    ary = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        minmax_scaling(ary, [1, 's2'])


def test_pandas_minmax_scaling():
    """Test minmax scaling numpy."""
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=['s1'])
    df['s2'] = s2

    df_out1 = minmax_scaling(df, ['s1', 's2'], min_val=0, max_val=1)
    df_out2 = minmax_scaling(df, ['s1', 's2'], min_val=50, max_val=100)

    ary_out1 = np.array([[0.0, 1.0],
                        [0.2, 0.8],
                        [0.4, 0.6],
                        [0.6, 0.4],
                        [0.8, 0.2],
                        [1.0, 0.0]])

    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)

    ary_out2 = np.array([[50.0, 100.0],
                        [60.0, 90.0],
                        [70.0, 80.0],
                        [80.0, 70.0],
                        [90.0, 60.0],
                        [100.0, 50.0]])

    np.testing.assert_allclose(df_out2, ary_out2, rtol=1e-03)


def test_numpy_minmax_scaling():
    """Test minmax scaling numpy."""
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])
    with pytest.raises(AttributeError):
        minmax_scaling(ary, [0, 1], min_val=0, max_val=1)
