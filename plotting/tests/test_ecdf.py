# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from mlearner.data import iris_data
from mlearner.plotting import ecdf


def test_threshold():

    X, y = iris_data()
    ax, threshold, count = ecdf(x=X[:, 0],
                                x_label='sepal length (cm)',
                                percentile=0.8)
    assert threshold == 6.5
    assert count == 120
