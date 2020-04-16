# Sebastian Raschka 2014-2020
# mlearn Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlearn.data import iris_data
from mlearn.data import wine_data
from mlearn.data import autompg_data
from mlearn.data import mnist_data
from mlearn.data import boston_housing_data
from mlearn.data import three_blobs_data


def test_import_wine_data():
    X, y = wine_data()

    assert(X.shape[0] == 178)
    assert(X.shape[1] == 13)
    assert(y.shape[0] == 178)


def test_import_iris_data():
    X, y = iris_data()
    assert(X.shape[0] == 150), X.shape
    assert(X.shape[1] == 4)
    assert(y.shape[0] == 150)


def test_import_autompg_data():
    X, y = autompg_data()
    assert(X.shape[0] == 392)
    assert(X.shape[1] == 8)
    assert(y.shape[0] == 392)


def test_import_mnist_data():
    X, y = mnist_data()
    assert(X.shape[0] == 5000)
    assert(X.shape[1] == 784)
    assert(y.shape[0] == 5000)


def test_import_three_blobs_data():
    X, y = three_blobs_data()
    assert(X.shape[0] == 150), X.shape
    assert(X.shape[1] == 2)
    assert(y.shape[0] == 150)


def test_import_boston_housing():
    X, y = boston_housing_data()
    assert(X.shape[0] == 506)
    assert(X.shape[1] == 13)
    assert(y.shape[0] == 506)
