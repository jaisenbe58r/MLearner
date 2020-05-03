"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import os
import pandas as pd

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "wine.csv")


def wine_data():
    """
    Wine dataset.

    Source: https://archive.ics.uci.edu/ml/datasets/Wine
    Number of samples: 178
    Class labels: {0, 1, 2}, distribution: [59, 71, 48]

    Data Set Information:

    These data are the results of a chemical analysis of wines grown
    in the same region in Italy but derived from three different cultivars.
    The analysis determined the quantities of 13 constituents found in each
    of the three types of wines.

    The attributes are (dontated by Riccardo Leardi, riclea@anchem.unige.it)

        - 1) Alcohol
        - 2) Malic acid
        - 3) Ash
        - 4) Alcalinity of ash
        - 5) Magnesium
        - 6) Total phenols
        - 7) Flavanoids
        - 8) Nonflavanoid phenols
        - 9) Proanthocyanins
        - 10) Color intensity
        - 11) Hue
        - 12) OD280/OD315 of diluted wines
        - 13) Proline

    In a classification context, this is a well posed problem with "well behaved"
    class structures. A good data set for first testing of a new classifier,
    but not very challenging.

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 178 wine samples as rows
        and 13 feature columns.
        y is a 1-dimensional array of the 3 class labels 0, 1, 2

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/wine_data

    adapted from
    https://github.com/rasbt/mlxtend/blob/master/mlxtend/data/wine.py
    Author: Sebastian Raschka <sebastianraschka.com>
    License: BSD 3 clause

    """
    data_csv = pd.read_csv(DATA_PATH, delimiter=',')
    X = data_csv[data_csv.columns[:-1]]
    y = data_csv[data_csv.columns[-1]].astype(int)
    return X, y
