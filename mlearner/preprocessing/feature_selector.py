
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):

    """This transformer select features.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FeatureSelector/

    """
    def __init__(self, columns=None, random_state=99):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make a replace missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self

        """
        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        if self.columns is None:
            self.columns = X.columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))
        self._fitted = True
        return self

    def transform(self, X):
        """this transformer handles missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns replaced.

        """
        if not hasattr(self, "_fitted"):
            raise AttributeError("FeatureSelector has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        return X[self.columns]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class CopyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, prefix=""):
        """Ini copy features."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        self.name = prefix

    def fit(self, X, y=None):
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
        self._fitted = True

        return self

    def transform(self, X):
        if not hasattr(self, "_fitted"):
            raise AttributeError("CopyFeatures has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        self.X_transform = X.copy()
        for i in self.columns:
            name_col = self.name + "_" + str(i)
            self.X_transform[name_col] = X[i].values

        return self.X_transform


class DropFeatures(BaseEstimator, TransformerMixin):

    """This transformer drop features.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/DropFeatures/

    """
    def __init__(self, columns_drop=None, random_state=99):
        """Init replace missing values."""
        if columns_drop is not None:
            if isinstance(columns_drop, list) or isinstance(columns_drop, tuple):
                self.columns_drop = columns_drop
            else:
                raise NameError("Invalid type {}".format(type(columns_drop)))
        else:
            self.columns_drop = columns_drop

        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make a replace missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self

        """
        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        if self.columns_drop is None:
            self.columns_drop = X.columns_drop

        _lista = [i for i in self.columns_drop if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))
        self._fitted = True
        return self

    def transform(self, X):
        """this transformer handles missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns replaced.

        """
        if not hasattr(self, "_fitted"):
            raise AttributeError("DropFeatures has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        return X.drop(self.columns_drop, axis=1)
