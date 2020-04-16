# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from mlearner.utils import assert_raises


def test_without_message():
    def my_func():
        raise AttributeError
    assert_raises(AttributeError, func=my_func, message=None)


def test_with_message():
        def my_func():
            raise AttributeError('Failed')
        assert_raises(AttributeError, func=my_func, message='Failed')
