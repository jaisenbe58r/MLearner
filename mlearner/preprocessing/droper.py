"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) - set(self.drop))]