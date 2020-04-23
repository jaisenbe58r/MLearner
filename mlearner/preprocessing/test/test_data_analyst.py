
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import pytest
from mlearner.preprocessing import DataAnalyst
import matplotlib
matplotlib.use('Template')

data = pd.DataFrame({"a": [0., 1., 1., 0., 1., 1.], "b": [10, 11, 12, 13, 11, 100], "c": ["OK", "OK", "NOK", "OK", "OK", "NOK"]})
col = ["a", "b"]


"""
DATA ANALYST -- BOXPLOT
"""


def test_init_type_data():
    with pytest.raises(TypeError):
        DataAnalyst(np.array(data))


def test_boxplot_features_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.boxplot(features=np.array(col))


def test_boxplot_features_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=[])


def test_boxplot_features_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=["d"])


def test_boxplot_features_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=["c"], target=["a"])


def test_boxplot_feature_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.boxplot(features=["a"], target=np.array(col))


def test_boxplot_feature_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["d"])


def test_boxplot_target_error_null2():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=[])


def test_boxplot_target_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["a", "b"])


def test_boxplot_target_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col)


def test_boxplot_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.boxplot(features=col, target=["c"], save_image=True, path="/invalid")


def test_boxplot_display_image():
    da = DataAnalyst(data)
    da.boxplot(features=["a", "b"], target=["c"], display=True)


def test_boxplot_features_none():
    da = DataAnalyst(data)
    da.boxplot(target=["c"], display=True)


"""
DATA ANALYST -- dISPERSION
"""


def test_dispersion_categoria_features_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.dispersion_categoria(features=np.array(col))


def test_dispersion_categoria_features_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=[])


def test_dispersion_categoria_features_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=["d"])


def test_dispersion_categoria_features_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=["c"], target=["a"])


def test_dispersion_categoria_target_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.dispersion_categoria(features=["a"], target=np.array(col))


def test_dispersion_categoria_target_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=col, target=["d"])


def test_dispersion_categoria_target_error_null2():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=col, target=[])


def test_dispersion_categoria_target_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=col, target=["a", "b"])


def test_dispersion_categoria_target_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=col)


def test_dispersion_categoria_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.dispersion_categoria(features=col, target=["c"], save_image=True, path="/incorrect")


def test_dispersion_categoria_save_image():
    da = DataAnalyst(data)
    da.dispersion_categoria(features=["a", "b"], target=["c"], display=True)


def test_dispersion_categoria_save_image1():
    da = DataAnalyst(data)
    da.dispersion_categoria(target=["c"], display=True)


"""
DATA ANALYST -- PAIRPLOT
"""


def test_sns_jointplot_feature1_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_jointplot(feature1=np.array(["a"]), feature2=["b"], target=["c"], categoria1="OK")


def test_sns_jointplot_feature1_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=[], feature2=["b"], target=["c"], categoria1="OK")


def test_sns_jointplot_feature1_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["d"], feature2=["b"], target=["c"], categoria1="OK")


def test_sns_jointplot_feature1_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["c"], feature2=["b"], target=["c"], categoria1="OK")


def test_sns_jointplot_feature2_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_jointplot(feature1=["a"], feature2=np.array(["b"]), target=["c"], categoria1="OK")


def test_sns_jointplot_feature2_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=[], target=["c"], categoria1="OK")


def test_sns_jointplot_feature2_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["d"], target=["c"], categoria1="OK")


def test_sns_jointplot_feature2_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["c"], target=["c"], categoria1="OK")


def test_sns_jointplot_target_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=np.array(["c"]), categoria1="OK")


def test_sns_jointplot_target_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=[], categoria1="OK")


def test_sns_jointplot_target_error_not_include():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["d"], categoria1="OK")


def test_sns_jointplot_target_error_none():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], categoria1="OK")


def test_sns_jointplot_categoria1_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=np.array(["OK"]))


def test_sns_jointplot_categoria1_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=[])


def test_sns_jointplot_categoria1_error_max():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK", "OK"])


def test_sns_jointplot_categoria1_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OKNOK"])


def test_sns_jointplot_categoria1_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=None)


def test_sns_jointplot_categoria2_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=np.array(["OK"]))


def test_sns_jointplot_categoria2_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=[])


def test_sns_jointplot_categoria2_error_max():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=["OK", "OK"])


def test_sns_jointplot_categoria2_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=["OKNOK"])


def test_sns_jointplot_categoria2_noerror_None():
    da = DataAnalyst(data)
    da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=None)


def test_sns_jointplot_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=["NOK"], save_image=True, path="/invalid")


def test_sns_jointplot_display():
    da = DataAnalyst(data)
    da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"], categoria2=["NOK"], display=True)


def test_sns_jointplot_test1():
    da = DataAnalyst(data)
    da.sns_jointplot(feature1=["a"], feature2=["b"], target=["c"], categoria1=["OK"])


"""
DATA ANALYST -- PAIRPLOT
"""


def test_sns_pairplot_features_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_pairplot(features=np.array(col))


def test_sns_pairplot_features_error_empty():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=[])


def test_sns_pairplot_features_error_not_included():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=["d"])


def test_sns_pairplot_features_error_object():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=["c"], target=["a"])


def test_sns_pairplot_target_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.sns_pairplot(features=["a"], target=np.array(col))


def test_sns_pairplot_target_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=col, target=["d"])


def test_sns_pairplot_target_error_null2():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=col, target=[])


def test_sns_pairplot_target_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=col, target=["a", "b"])


def test_sns_pairplot_target_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=col)


def test_sns_pairplot_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.sns_pairplot(features=col, target=["c"], save_image=True, path="/incorrect")


def test_sns_pairplot_save_image():
    da = DataAnalyst(data)
    da.sns_pairplot(features=["a", "b"], target=["c"], display=True)


def test_sns_pairplot_save_image1():
    da = DataAnalyst(data)
    da.sns_pairplot(target=["c"], display=True)


"""
DATA ANALYST -- Distribution targets
"""


def test_distribution_targets_target_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.distribution_targets(target=np.array(col))


def test_distribution_targets_target_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.distribution_targets(target=["d"])


def test_distribution_targets_target_error_null2():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.distribution_targets(target=[])


def test_distribution_targets_target_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.distribution_targets(target=["a", "b"])


def test_distribution_targets_target_error_None():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.distribution_targets()


def test_distribution_targets_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.distribution_targets(target=["c"], save_image=True, path="/incorrect")


def test_distribution_targets_save_image():
    da = DataAnalyst(data)
    da.distribution_targets(target=["c"], display=True)


def test_distribution_targets_save_image1():
    da = DataAnalyst(data)
    da.distribution_targets(target=["c"], display=False)


"""
DATA ANALYST -- corr_matrix
"""


def test_corr_matrix_feature_error_type():
    da = DataAnalyst(data)
    with pytest.raises(TypeError):
        da.corr_matrix(features=np.array(["a", "b"]))


def test_corr_matrix_feature_error_null():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.corr_matrix(features=["d"])


def test_corr_matrix_feature_error_null2():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.corr_matrix(features=[])


def test_corr_matrix_feature_error_only():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.corr_matrix(features=["c"])


def test_corr_matrix_feature_error_None():
    da = DataAnalyst(data)
    da.corr_matrix()


def test_corr_matrix_incorrect_path():
    da = DataAnalyst(data)
    with pytest.raises(NameError):
        da.corr_matrix(features=["a", "b"], save_image=True, path="/incorrect")


def test_corr_matrix_save_image():
    da = DataAnalyst(data)
    da.corr_matrix(features=["a", "b"], display=True)


def test_corr_matrix_save_image1():
    da = DataAnalyst(data)
    da.corr_matrix(features=["a", "b"], display=False)
