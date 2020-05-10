
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer <www.linkedin.com/in/jaisenbe>
License: MIT
"""


from sklearn.preprocessing import LabelEncoder, StandardScaler
from .scaling import minmax_scaling
from .mean_centering import MeanCenterer
from .droper import FeatureDropper, DropOutliers
from .replace_na import FillNaTransformer_forward, FillNaTransformer_backward, FillNaTransformer_value, FillNaTransformer_all, FillNaTransformer_any, FillNaTransformer_median, FillNaTransformer_mean, FillNaTransformer_idmax
from .log_skewed import FixSkewness
from .one_hot_encoder import OneHotEncoder, CategoricalEncoder
from .replace_categorical import ReplaceTransformer, ReplaceMulticlass
from .extract_target import ExtractCategories
from .base_preprocess import DataExploratory, DataAnalyst
from .feature_selector import FeatureSelector, DataFrameSelector, DropFeatures, CopyFeatures
from .reduce_feature import PCA_selector, LDA_selector, PCA_add, LDA_add
from .perfomance_transformers import OrientationClassTransformer, MFD_OrientationClassTransformer
from .perfomance_transformers import ClassTransformer_value, Keep

__all__ = ["minmax_scaling", "MeanCenterer", "FeatureDropper",
            "FillNaTransformer_median", "FillNaTransformer_mean",
            "FillNaTransformer_idmax", "FillNaTransformer_any",
            "FillNaTransformer_all", "FillNaTransformer_value",
            "FillNaTransformer_backward", "FillNaTransformer_forward",
            "FixSkewness", "OneHotEncoder", "DropOutliers", "ReplaceTransformer",
            "ReplaceMulticlass", "ExtractCategories", "DataExploratory",
            "DataAnalyst", "LabelEncoder", "StandardScaler",
            "FeatureSelector", "DataFrameSelector", "CategoricalEncoder",
            "PCA_selector", "LDA_selector", "PCA_add", "LDA_add",
            "DropFeatures", "CopyFeatures",
            "OrientationClassTransformer", "MFD_OrientationClassTransformer"
            "ClassTransformer_value", "Keep"]
