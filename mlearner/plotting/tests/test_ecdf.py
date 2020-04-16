# Sebastian Raschka 2014-2020
# mlearner Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlearner.data import iris_data
from mlearner.plotting import ecdf


def test_threshold():

    X, y = iris_data()
    ax, threshold, count = ecdf(x=X[:, 0],
                                x_label='sepal length (cm)',
                                percentile=0.8)
    assert threshold == 6.5
    assert count == 120
