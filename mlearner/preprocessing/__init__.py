
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer <www.linkedin.com/in/jaisenbe>
License: MIT
"""

from .scaling import minmax_scaling
from .mean_centering import MeanCenterer
from .droper import FeatureDropper
from .replace_na import FillNaTransformer_forward, FillNaTransformer_backward, FillNaTransformer_value, FillNaTransformer_all, FillNaTransformer_any, FillNaTransformer_median, FillNaTransformer_mean, FillNaTransformer_idmax
from .log_skewed import FixSkewness
from .one_hot_encoder import OneHotEncoder

__all__ = ["minmax_scaling", "MeanCenterer", "FeatureDropper",
            "FillNaTransformer_median", "FillNaTransformer_mean",
            "FillNaTransformer_idmax", "FillNaTransformer_any",
            "FillNaTransformer_all", "FillNaTransformer_value",
            "FillNaTransformer_backward", "FillNaTransformer_forward",
            "FixSkewness", "OneHotEncoder"]
