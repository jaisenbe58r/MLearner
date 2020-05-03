
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class PCA_selector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_components=2, random_state=99):
        """Init log PCA_selector."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        if n_components is not None:
            if isinstance(n_components, int):
                self.n_components = n_components
            else:
                raise TypeError("Invalid type {}".format(type(n_components)))
        else:
            self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Selecting PCA columns from the dataset.

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

        self._fitted = True

        return self

    def transform(self, X):
        """Trransformer applies PCA.

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

        if not hasattr(self, "_fitted"):
            raise AttributeError("PCA_selector has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        self.X_std = StandardScaler().fit_transform(X[self.columns])

        return PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(self.X_std)


class LDA_selector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, random_state=99):
        """Init log LDA_selector."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns
        self.random_state = random_state

    def fit(self, X, y):
        """Selecting LDA columns from the dataset.

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

        if not isinstance(y, pd.core.frame.DataFrame) and not isinstance(y, pd.core.series.Series):
            raise NameError("Invalid type {}".format(type(y)))

        self.LDA = LinearDiscriminantAnalysis().fit(X[self.columns], y)

        return self

    def transform(self, X):
        """Trransformer applies LDA.

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

        if not hasattr(self, "LDA"):
            raise AttributeError("LDA_selector has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        return self.LDA.transform(X[self.columns])


class PCA_add(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_components=2, PCA_name=None, random_state=99):
        """Init log PCA_add."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        if n_components is not None:
            if isinstance(n_components, int):
                self.n_components = n_components
            else:
                raise TypeError("Invalid type {}".format(type(n_components)))
        else:
            self.n_components = n_components

        self.PCA_name = PCA_name
        self.random_state = random_state

    def fit(self, X, y=None):
        """Selecting PCA columns from the dataset.

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

        self._fitted = True

        return self

    def _add_dataframe(self, Y):

        df = pd.DataFrame()
        for i in range(Y.shape[1]):
            nombre = str(self.PCA_name) + "_PCA_" + str(i+1)
            df[nombre] = Y[:, i]
        return df

    def _concat_dataframe(self, X_transf, df):

        for i in df.columns.tolist():
            X_transf[i] = df[i].values
        return X_transf

    def transform(self, X):
        """Trransformer applies PCA.

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

        if not hasattr(self, "_fitted"):
            raise AttributeError("PCA_selector has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X_transform = X.copy()
        self.X_std = StandardScaler().fit_transform(X_transform[self.columns])
        X_PCA = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(self.X_std)
        df_PCA = self._add_dataframe(X_PCA)

        return self._concat_dataframe(X_transform, df_PCA)


class LDA_add(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, LDA_name=None, random_state=99):
        """Init log LDA_selector."""
        if columns is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise TypeError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

        self.LDA_name = LDA_name
        self.random_state = random_state

    def fit(self, X, y=None):
        """Selecting LDA columns from the dataset.

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

        if not isinstance(y, pd.core.frame.DataFrame) and not isinstance(y, pd.core.series.Series):
            raise NameError("Invalid type {}".format(type(y)))

        self.LDA = LinearDiscriminantAnalysis().fit(X[self.columns], y)

        return self

    def _add_dataframe_LDA(self, Y):

        df_LDA = pd.DataFrame()
        for i in range(Y.shape[1]):
            nombre = str(self.LDA_name) + "_LDA_" + str(i+1)
            df_LDA[nombre] = Y[:, i]
        return df_LDA

    def _concat_dataframe(self, X_transf, df_LDA):

        for i in df_LDA.columns.tolist():
            X_transf[i] = df_LDA[i].values
        return X_transf

    def transform(self, X):
        """Trransformer applies LDA.

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

        if not hasattr(self, "X_LDA"):
            raise AttributeError("LDA_selector has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X_transform = X.copy()
        X_LDA = self.LDA.transform(X_transform)
        df_LDA = self._add_dataframe_LDA(X_LDA)

        return self._concat_dataframe(X_transform, df_LDA)
