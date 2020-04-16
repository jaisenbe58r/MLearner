# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from mlearner.utils import Counter


def test_counter():
    cnt = Counter()
    for i in range(20):
        cnt.update()
    assert cnt.curr_iter == 20
