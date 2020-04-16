# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from mlearner.plotting import checkerboard_plot
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')


def test_runs():

    ary = np.random.random((6, 4))
    checkerboard_plot(ary,
                      col_labels=['abc', 'def', 'ghi', 'jkl'],
                      row_labels=['sample %d' % i for i in range(1, 6)],
                      cell_colors=['skyblue', 'whitesmoke'],
                      font_colors=['black', 'black'],
                      figsize=(5, 5))
