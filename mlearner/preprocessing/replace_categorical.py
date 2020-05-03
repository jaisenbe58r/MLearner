
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..externals.estimator_checks import check_is_fitted


class ReplaceTransformer(BaseEstimator, TransformerMixin):

    """This transformer replace some values with others.

    Attributes
    ----------
    columns: `list` of columns to transformer [n_columns]

    mapping: dict`, for example:
        ```
        mapping = {"yes": 1, "no": 0}
        ```

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/ReplaceTransformer/

    """
    def __init__(self, columns=None, mapping=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        if mapping is not None:
            if isinstance(mapping, dict):
                self.mapping = mapping
            else:
                raise TypeError("Invalid type {}".format(type(mapping)))
        else:
            raise NameError("Unspecified mapping")

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make a replace values.

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
            raise TypeError("Invalid type {}".format(type(X)))

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

        if self.mapping is not None:
            _keys = list(self.mapping)
            for col in self.columns:
                _lista = [i for i in _keys if i not in X[col].unique().tolist()]
                if len(_lista) > 0:
                    raise NameError("The Keys {} no exist on column {}".format(_lista, col))

        self._fitted = True
        return self

    def transform(self, X):
        """Gets the columns to make a replace values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transform : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns replaced.

        """
        check_is_fitted(self, '_fitted')

        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Invalid type {}".format(type(X)))

        X_transform = X.copy()
        for f in self.columns:
            X_transform[f] = X_transform[f].replace(self.mapping)

        return X_transform


class ReplaceMulticlass(BaseEstimator, TransformerMixin):

    """This transformer replace some categorical values with others.

    Attributes
    ----------
    columns: `list` of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/ReplaceMulticlass/

    """
    def __init__(self, columns=None):
        """Init replace categorical values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make a replace values.

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
            raise TypeError("Invalid type {}".format(type(X)))

        if self.columns is None:
            self.columns = X.columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

        self._fitted = True
        return self

    def transform(self, X):
        """Gets the columns to make a replace to categorical values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transform : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns replaced.

        """
        check_is_fitted(self, '_fitted')

        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Invalid type {}".format(type(X)))

        X_transform = X.copy()
        for f in self.columns:
            _unic = X[f].unique().tolist()
            _remp = np.arange(0, len(_unic)).tolist()
            X_transform[f] = X_transform[f].replace(_unic, _remp)

        return X_transform
