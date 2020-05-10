
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from mlearner.utils import ParamsManager
# from mlearner.models import modelLightBoost

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/classifier/config/models.json"


class FeatureSelection(object):
    def __init__(self, random_state=99):
        """
        Inicializacion e la clase de Seleccion de rasgos
        """
        self.random_state = random_state
        self.manager_models = ParamsManager(param_file, key_read="Models")

    def cor_pearson(self, X, y, k='all'):
        """
        Pearson Correlation
            Normalization: no
            Impute missing values: yes
        """
        if not k == 'all':
            if k > len(X.columns.tolist()):
                raise NameError("Numero de features seleccionas (k) mayor a features totales")

        feature_name = X.columns.tolist()
        cor_list = []
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        if k == 'all':
            cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[:]].columns.tolist()

        else:
            it = k*-1
            cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[it:]].columns.tolist()

        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]

        print(str(len(cor_feature)), 'selected features')

        return cor_support, cor_feature

    def chi2(self, X, y, k='all'):
        """
        Chi-2
            Normalization: MinMaxScaler (values should be bigger than 0)
            Impute missing values: yes
        """
        if not k == 'all':
            if k > len(X.columns.tolist()):
                raise NameError("Numero de features seleccionas (k) mayor a features totales")

        feature_name = X.columns.tolist()
        X_norm = MinMaxScaler().fit_transform(X)
        chi_selector = SelectKBest(chi2, k=k)
        chi_selector.fit(X_norm, y)

        _chi_support = chi_selector.get_support()
        chi_feature = X.loc[:, _chi_support].columns.tolist()
        print(str(len(chi_feature)), 'selected features')

        chi_support = [True if i in chi_feature else False for i in feature_name]

        return chi_support, chi_feature, chi_selector

    def wrapper(self, X, y, k='all'):
        """
        Wrapper
        documentation for RFE: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

            Normalization: depend on the used model; yes for LR
            Impute missing values: depend on the used model; yes for LR
        """
        X_norm = MinMaxScaler().fit_transform(X)

        if not k == 'all':
            if k > len(X.columns.tolist()):
                raise NameError("Numero de features seleccionas (k) mayor a features totales")
        else:
            k = len(X.columns.tolist())

        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=k, step=10, verbose=0)
        rfe_selector.fit(X_norm, y)

        rfe_support = rfe_selector.get_support()
        rfe_feature = X.loc[:, rfe_support].columns.tolist()
        print(str(len(rfe_feature)), 'selected features')

        return rfe_support, rfe_feature, rfe_selector

    def embeded(self, X, y):
        """
        Embeded
            documentation for SelectFromModel: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html ### 3.1 Logistics Regression L1 Note
            Normalization: Yes
            Impute missing values: Yes
        """
        X_norm = MinMaxScaler().fit_transform(X)

        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), '1.25*median')
        embeded_lr_selector.fit(X_norm, y)

        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()
        print(str(len(embeded_lr_feature)), 'selected features')

        return embeded_lr_support, embeded_lr_feature, embeded_lr_selector

    def RandomForest(self, X, y, n_estimators=100):
        """
        Random Forest

            Normalization: No
            Impute missing values: Yes
        """
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators), threshold='1.25*median')
        embeded_rf_selector.fit(X, y)

        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
        print(str(len(embeded_rf_feature)), 'selected features')

        return embeded_rf_support, embeded_rf_feature, embeded_rf_selector

    def LightGBM(self, X, y, n_estimators=100):
        """
        LightGBM

            Normalization: No
            Impute missing values: No
        """
        from lightgbm import LGBMClassifier
        lgbc = LGBMClassifier(n_estimators=n_estimators)
        # lgbc = modelLightBoost()

        embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
        embeded_lgb_selector.fit(X, y)

        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = X.loc[:, embeded_lgb_support].columns.tolist()
        print(str(len(embeded_lgb_feature)), 'selected features')

        return embeded_lgb_support, embeded_lgb_feature, embeded_lgb_selector

    def transform_data(self, X, selector, features):
        """
        Transformar el conjunto de entrenamiento al nuevo esquema proporcionado por el selector
        """
        dff = selector.transform(X)
        df_X = pd.DataFrame(columns=features, data=dff)

        return df_X

    def Summary(self, X, y, k="all", cor_pearson=True, chi2=True, wrapper=True, embeded=True,
                RandomForest=True, LightGBM=True):
        """
        Resumen de la seleccion de caracteristicas.
        """
        feature_name = X.columns.tolist()
        pd.set_option('display.max_rows', None)

        # put all selection together
        feature_selection_df = pd.DataFrame({'Feature': feature_name})

        if cor_pearson:
            cor_support, _ = self.cor_pearson(X, y, k)
            feature_selection_df['Pearson'] = cor_support
        if chi2:
            chi_support, _, _ = self.chi2(X, y, k)
            feature_selection_df['Chi2'] = chi_support
        if wrapper:
            rfe_support, _, _ = self.wrapper(X, y, k)
            feature_selection_df['RFE'] = rfe_support
        if embeded:
            embeded_lr_support, _, _ = self.embeded(X, y)
            feature_selection_df['Logistics'] = embeded_lr_support
        if RandomForest:
            embeded_rf_support, _, _ = self.RandomForest(X, y)
            feature_selection_df['Random Forest'] = embeded_rf_support
        if LightGBM:
            embeded_lgb_support, _, _ = self.LightGBM(X, y)
            feature_selection_df['LightGBM'] = embeded_lgb_support

        # count the selected times for each feature
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        # display the top 100
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df)+1)
        feature_selection_df.head(100)

        return feature_selection_df
