
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanCenterer(BaseEstimator, TransformerMixin):

    """Column centering of pandas Dataframeself.

    Attributes
    ----------
    col_means:  numpy.ndarray [n_columns] or pandas [n_columns]
    mean values for centering after fitting the MeanCenterer object.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/MeanCenterer/

    """
    def __init__(self):
        """Init Mean Center."""
        pass

    def fit(self, X, y=None):
        """Gets the column means for mean centering.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self
        """
        if isinstance(X, pd.core.frame.DataFrame):
            try:
                _test = X.astype(np.float32)
                del(_test)
            except ValueError:
                raise NameError("Null or categorical variables are not allowed: {}".format(X.dtypes))

            self.col_means = X.mean()

        elif isinstance(X, np.ndarray):
            try:
                _test = np.mean(X)
            except TypeError:
                raise NameError("Null or categorical variables are not allowed")

            _X_fl = X.astype('float')
            self.col_means = _X_fl.mean(axis=0)

        else:
            raise NameError("Invalid type {}".format(type(X)))

        print(self.col_means)

        return self

    def transform(self, X):
        """Centers a pandas.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transform : {DAtaframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns centered.

        """
        if not hasattr(self, "col_means"):
            raise AttributeError("MeanCenterer has not been fitted, yet.")

        if isinstance(X, pd.core.frame.DataFrame):
            X_transform = X.copy()
            for i in range(X.shape[1]):
                X_transform[i] = np.apply_along_axis(func1d=lambda x: (x - X.mean()),
                                                        axis=1, arr=X)

        elif isinstance(X, np.ndarray):
            X_transform = np.apply_along_axis(func1d=lambda x: x - self.col_means,
                                                axis=1, arr=X)
        else:
            raise NameError("Invalid type {}".format(type(X)))
        return X_transform
