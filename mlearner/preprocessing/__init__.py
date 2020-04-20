
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from .scaling import minmax_scaling
from .mean_centering import MeanCenterer
from .droper import FeatureDropper

__all__ = ["minmax_scaling", "MeanCenterer", "FeatureDropper"]
