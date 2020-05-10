"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OrientationClassTransformer(BaseEstimator, TransformerMixin):
    """Transformer Orientation."""
    def __init__(self, columns, name="OCT", a=135, b=45):
        self.columns = columns
        self.name = name
        self.a, self.b = a, b

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        func = lambda x: 0 if (x <= self.a and x >= self.b) else 1
        d = map(func, X[self.columns].values)

        return np.fromiter(d, dtype=np.int32).reshape(-1, 1)


class MFD_OrientationClassTransformer(BaseEstimator, TransformerMixin):
    """Transformer MFD Orientation."""
    def __init__(self, columns, name="MFDOCT", a=120, b=60, c=30, d=150):
        self.columns = columns
        self.name = name
        self.a, self.b = a, b
        self.c, self.d = c, d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        func = lambda x: 0 if (x <= self.a and x >= self.b) else ( 1 if (x < self.c or x > self.d) else 2)
        d = map(func, X[self.columns].values)

        return np.fromiter(d, dtype=np.int32).reshape(-1, 1)


class ClassTransformer_value(BaseEstimator, TransformerMixin):
    def __init__(self, columns, name="A/AH_cat", value=100):
        self.columns = columns
        self.name = name
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        func = lambda x: 0 if x == self.value else 1
        d = map(func, X[self.columns].values)

        return np.fromiter(d, dtype=np.int32).reshape(-1, 1)


class Keep(BaseEstimator, TransformerMixin):
    """Mantener columnas."""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X