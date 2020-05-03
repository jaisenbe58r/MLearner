"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from mlearner.data import wine_data


def test_import_wine():
    """Test import dataset wine."""
    X, y = wine_data()

    assert(X.shape[1] == 13)
    assert(len(y.unique().tolist()) == 3)
    assert(X.shape[0] == y.shape[0])
