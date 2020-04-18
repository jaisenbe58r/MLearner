# Jaime Sendra Berenguer-2020.
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
# License: MIT

import pandas as pd
import numpy as np


def minmax_scaling(array, columns, min_val=0, max_val=1):

    """In max scaling of pandas' DataFrames.

    Parameters
    --------

    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].
    columns : array-like, shape = [n_columns]
        Array-like with column names, e.g., ['col1', 'col2', ...]
        or column indices [0, 2, 4, ...]
    min_val : `int` or `float`, optional (default=`0`)
        minimum value after rescaling.
    max_val : `int` or `float`, optional (default=`1`)
        maximum value after rescaling.

    Returns
    --------

    df_new : pandas DataFrame object.
        Copy of the array or DataFrame with rescaled columns.

    Examples
    --------

    For usage examples, please see
    http://jaisenbe58r.github.io/mlearner/user_guide/preprocessing/minmax_scaling/.

    """
    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas'
                             'DataFrame or NumPy array')

    numerator = ary_newt[:, columns] - ary_newt[:, columns].min(axis=0)
    denominator = (ary_newt[:, columns].max(axis=0) -
                   ary_newt[:, columns].min(axis=0))
    ary_newt[:, columns] = numerator / denominator

    if not min_val == 0 and not max_val == 1:
        ary_newt[:, columns] = (ary_newt[:, columns] *
                                (max_val - min_val) + min_val)
    return ary_newt[:, columns]
