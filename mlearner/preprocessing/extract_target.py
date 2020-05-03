
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..externals.estimator_checks import check_is_fitted


class ExtractCategories(BaseEstimator, TransformerMixin):

    """This transformer filters the selected dataset categories.

    Attributes
    ----------
    categories: `list` of categories that you want to keep.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/ReplaceTransformer/

    """
    def __init__(self, categories=None, target=None):
        """Init filter categories"""
        if categories is not None:
            if isinstance(categories, list) or isinstance(categories, tuple):
                if len(categories) == 0:
                    raise NameError("Empty category List")
                else:
                    self.categories = categories
            else:
                raise TypeError("Invalid type {}".format(type(categories)))
        else:
            self.categories = categories

        if target is not None:
            if isinstance(target, list) or isinstance(target, tuple):
                if len(target) == 0:
                    raise NameError("Empty category List")
                elif len(target) > 1:
                    raise NameError("Only one category column can be selected")
                else:
                    self.target = target[0]
            else:
                raise TypeError("Invalid type {}".format(type(target)))
        else:
            self.target = target

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make filters the selected dataset categories.

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

        if self.target is None:
            self.target = X.columns.tolist()[-1]
        else:
            if self.target not in X.columns.tolist():
                print(X.columns.tolist())
                raise NameError("Target '{}' not included in dataset columns".format(self.target))

        if self.categories is None:
            self.categories = X[self.target].unique().tolist()
        else:
            for i in self.categories:
                if i not in X[self.target].unique().tolist():
                    raise NameError("Category '{}' not included in dataset targets".format(i))

        self._fitted = True
        return self

    def transform(self, X):
        """Gets the columns to make filters the selected dataset categories.

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
        X_transform = X[X[self.target] == self.categories[0]]
        if len(self.categories) > 1:
            for i in range(len(self.categories)-1):
                X_transform_aux = X[X[self.target] == self.categories[i+1]]
                X_transform = pd.merge(left=X_transform, right=X_transform_aux, how="outer")

        return X_transform

