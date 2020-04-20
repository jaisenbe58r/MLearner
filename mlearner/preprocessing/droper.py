"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# https://www.kaggle.com/funkyfrankie/sklearn-pipelines-and-transformers


class FeatureDropper(BaseEstimator, TransformerMixin):

    """Column drop according to the selected feature.

    Attributes
    ----------
    drop: list of features to drop [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FeatureDropper/

    """

    def __init__(self, drop=[]):
        """Init Feature droped."""
        if isinstance(drop, list):
            self.drop = drop
        else:
            raise NameError("Invalid type {}".format(type(drop)))

    def fit(self, X, y=None, **fit_params):
        """Gets the columns that not drop.

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
            self.col_drop = self.drop
        else:
            raise NameError("Invalid type {}".format(type(X)))

        return self

    def transform(self, X, **fit_params):
        """Features drop.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transform : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns dropped.

        """
        if not hasattr(self, "col_drop"):
            raise AttributeError("FeatureDropper has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        return X.drop(self.col_drop, axis=1)
