
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from mlearner.preprocessing import FillNaTransformer_any


data = pd.DataFrame({"a": [2, 2, 2, float('nan'), 4, 5], "b": ["OK", "OK", float('nan'), "OK", "OK", "NOK"]})
data_out = data.dropna(axis=0, how="any")


def test_any_tranf_equal():
    ft = FillNaTransformer_any()
    assert_frame_equal(ft.transform(data), data_out)


def test_any_invalid_transf_type():
    ft = FillNaTransformer_any()
    ft.fit(data)
    with pytest.raises(NameError):
        ft.transform(data.values)
