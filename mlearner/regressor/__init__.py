# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from .linear_regression import LinearRegression
from .stacking_regression import StackingRegressor
from .stacking_cv_regression import StackingCVRegressor

__all__ = ["LinearRegression", "StackingRegressor", "StackingCVRegressor"]
