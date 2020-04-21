
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):

    """This transformer applies One-Hot-Encoder to features.

    Attributes
    ----------
    numerical: pandas [n_columns].
        numerical columns to be treated as categorical.
    columns:  pandas [n_columns].
        columns to use (if None then all categorical variables are included).

    Examples
    --------
    For usage examples, please see:
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/OneHotEncoder/


    """
    def __init__(self, columns=None, numerical=[]):
        """Init OneHotEncoder."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        if numerical != []:
            if isinstance(numerical, list) or isinstance(numerical, tuple):
                self.numerical = numerical
            else:
                raise TypeError("Invalid type {}".format(type(numerical)))
        else:
            self.numerical = numerical

    def fit(self, X, y=None, **fit_params):
        """Selecting OneHotEncoder columns from the dataset.

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
            self.columns = X.select_dtypes(include=["object"]).columns.tolist()

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        self.columns += self.numerical
        # get all possible column values to filter not seen values
        self.allowed_columns = ["{}_{}".format(column, val) for column in self.columns for val in X[column].unique()]

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
            A copy of the input Dataframe with the columns encoder.

        """
        if not hasattr(self, "allowed_columns"):
            raise AttributeError("FixSkewness has not been fitted, yet.")

        if isinstance(X, pd.core.frame.DataFrame):
            X_transform = X.copy()
        else:
            raise NameError("Invalid type {}".format(type(X)))

        # cast numerical columns to strings.
        for col in X_transform[self.columns].select_dtypes(exclude=["object"]).columns:
            X_transform[col] = X_transform[col].astype('str')

        one_hots = pd.get_dummies(X_transform[self.columns], prefix=self.columns)
        missing_cols = set(self.allowed_columns) - set(one_hots.columns)

        for c in missing_cols:
            one_hots[c] = 0

        return pd.concat([X_transform.drop(self.columns, axis=1), one_hots.filter(self.allowed_columns)], axis=1)
