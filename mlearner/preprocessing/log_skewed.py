
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


class FixSkewness(BaseEstimator, TransformerMixin):

    """This transformer applies log to skewed features.

    Attributes
    ----------
    columns:  npandas [n_columns]

    Examples
    --------
    For usage examples, please see:
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FixSkewness/


    """
    def __init__(self, columns=None, drop=True):
        """Init log skewed."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns
        self.drop = drop

    def fit(self, X, y=None, **fit_params):
        """Selecting skewed columns from the dataset.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self
        """
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns

        if isinstance(X, pd.core.frame.DataFrame):
            try:
                _test = X[self.columns].astype(np.float32)
                del(_test)
            except ValueError:
                raise NameError("Null or categorical variables are not allowed: {}".format(X.dtypes))
        else:
            raise NameError("Invalid type {}".format(type(X)))

        skewness = X[self.columns].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        self.skew_features = skewness.index

        return self

    def transform(self, X):
        """Trransformer applies log to skewed features.

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
        if not hasattr(self, "skew_features"):
            raise AttributeError("FixSkewness has not been fitted, yet.")

        if isinstance(X, pd.core.frame.DataFrame):
            X_transform = X.copy()
            X_transform[self.skew_features] = np.log1p(X[self.skew_features])
        else:
            raise NameError("Invalid type {}".format(type(X)))

        if self.drop:
            X_drop = [i for i in X.columns.tolist() if i not in self.columns]
            X_transform = X_transform.drop(X_drop, axis=1)

        return X_transform
