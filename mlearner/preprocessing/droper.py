"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from ..externals.estimator_checks import check_is_fitted
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


class DropOutliers(BaseEstimator, TransformerMixin):

    """Drop Outliers from dataframe

    Attributes
    ----------
    features: `list`or `tuple
        list of features to drop outliers [n_columns]
    display: `boolean`
        Show histogram with changes made.

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/DropOutliers/

    """

    def __init__(self, features=[], display=False):
        """Init Feature droped."""
        if isinstance(features, list) or isinstance(features, tuple):
            self.features = features
        else:
            raise TypeError("Invalid type {}".format(type(features)))

        if isinstance(display, bool):
            self.display = display
        else:
            raise TypeError("Invalid type {}".format(type(display)))

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
        if self.features == []:
            self.features = X.select_dtypes(exclude=["object"]).columns.tolist()

        if isinstance(X, pd.core.frame.DataFrame):
            try:
                _test = X[self.features].astype(np.float32)
                del(_test)
            except ValueError:
                raise ValueError("Null or categorical variables are not allowed: {}".format(X.dtypes))
        else:
            raise TypeError("Invalid type {}".format(type(X)))

        self._fit = True

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
        check_is_fitted(self, '_fit')

        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("Invalid type {}".format(type(X)))

        if self.display:
            height, width = self._get_size_plot()
            figure, axs = plt.subplots(height, width, figsize=(width*3, height*3))
            ax = axs.flatten()

        data_transform = X.copy()
        for i in range(len(self.features)):
            _f = self.features[i]

            # Rango intercuartilico (Q3–Q1)
            IQR = data_transform[_f].quantile(0.75)-data_transform[_f].quantile(0.25)

            # se consideran atipicos los valores inferiores a Q1–1.5·IQR o superiores a Q3+1.5·IQR
            IQR_max = data_transform[_f].quantile(0.75) + 1.5*IQR
            IQR_min = data_transform[_f].quantile(0.25) - 1.5*IQR

            data_transform = data_transform[(data_transform[_f] > IQR_min) & (data_transform[_f] < IQR_max)]

            if self.display:
                ax[i].hist(data_transform[_f], density=True, color="g", alpha=0.7)
                ax[i].hist(X[_f], density=True, color="r", alpha=0.7)
                ax[i].set_title(_f)
                ax[i].legend(["Original", "Cleaned"])

        if self.display:
            figure.tight_layout()
            plt.show()

        return data_transform

    def _get_size_plot(self, width=4):
        """Size of plot."""
        check_is_fitted(self, '_fit')

        _n = len(self.features)

        if not _n <= width:
            width = 4
            nm = _n/width
        else:
            if _n > 1:
                width = _n
            else:
                width = 2
            nm = 0
        height = int(nm)+1

        return height, width
