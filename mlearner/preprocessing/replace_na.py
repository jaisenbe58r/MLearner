

"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FillNaTransformer_median(BaseEstimator, TransformerMixin):

    """This transformer handles missing values.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_median/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

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
            self.columns = X.select_dtypes(exclude=["object"]).columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

        self.train_median = X[self.columns].median()

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
        if not hasattr(self, "train_median"):
            raise AttributeError("FillNaTransformer_median has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X[self.columns] = X[self.columns].fillna(self.train_median)

        return X


class FillNaTransformer_mean(BaseEstimator, TransformerMixin):

    """This transformer handles missing values.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_mean/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

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
            self.columns = X.select_dtypes(exclude=["object"]).columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

        self.train_mean = X[self.columns].mean()

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
        if not hasattr(self, "train_mean"):
            raise AttributeError("FillNaTransformer_median has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X[self.columns] = X[self.columns].fillna(self.train_mean)

        return X


class FillNaTransformer_idmax(BaseEstimator, TransformerMixin):

    """This transformer handles missing values for idmax.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_idmax/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

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
            # self.columns = X.select_dtypes(exclude=["object"]).columnsX.select_dtypes(exclude=["object"]).columns
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
            raise AttributeError("FillNaTransformer_idmax has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        for col in self.columns:
            X[col] = X[col].fillna(X[col].value_counts().idxmax())

        return X


class FillNaTransformer_any(BaseEstimator, TransformerMixin):

    """This transformer delete row that there is some NaN.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_any/

    """
    def __init__(self):
        """Init replace missing values."""

    def fit(self, X, y=None, **fit_params):
        """Not implemented.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self

        """
        return self

    def transform(self, X):
        """This transformer delete row that there is some NaN

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
        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        self.columns = X.columns
        X.dropna(axis=0, how="any", inplace=True)
        # X.reset_index()
        # X.drop(["index"], axis=1)

        return X


class FillNaTransformer_all(BaseEstimator, TransformerMixin):

    """This transformer delete row that there is all NaN.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_all/

    """
    def __init__(self):
        """Init replace missing values."""

    def fit(self, X, y=None, **fit_params):
        """Not implemented.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self

        """
        return self

    def transform(self, X):
        """This transformer delete row that there is some NaN

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
        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        self.columns = X.columns
        X.dropna(axis=0, how="all", inplace=True)
        # X.reset_index()
        # X.drop(["index"], axis=1, inplace=True)

        return X


class FillNaTransformer_value(BaseEstimator, TransformerMixin):

    """This transformer handles missing values.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_value/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

    def fit(self, X, y=None, value=None, **fit_params):
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

        if value is not None:
            if (not isinstance(value, bool) and not isinstance(value, int)
                    and not isinstance(value, float) and not isinstance(value, str)):
                raise NameError("Type value not permited: {}".format(type(value)))
            else:
                self.value = value
        else:
            raise NameError("Not alowed value=None")

        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

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
        if not hasattr(self, "value"):
            raise AttributeError("FillNaTransformer_value has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X[self.columns] = X[self.columns].fillna(self.value)
        return X


class FillNaTransformer_backward(BaseEstimator, TransformerMixin):

    """This transformer handles missing values closer backward.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_backward/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

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
            raise AttributeError("FillNaTransformer_backward has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))
        X_transform = X.copy()
        X_transform[self.columns] = X[self.columns].fillna(method="bfill")
        return X_transform


class FillNaTransformer_forward(BaseEstimator, TransformerMixin):

    """This transformer handles missing values closer forward.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_forward/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

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
            raise AttributeError("FillNaTransformer_backward has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X_transform = X.copy()
        X_transform[self.columns] = X[self.columns].fillna(method="ffill")
        return X_transform
