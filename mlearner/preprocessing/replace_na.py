

"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FillNaTransformer_median(BaseEstimator, TransformerMixin):  

    """This transformer handles missing values.

    Attributes
    ----------
    columns: list of columns to transformer [n_columns]

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/FillNaTransformer_median/

    """
    def __init__(self, columns=None):
        """Init replace missing values."""
        if column is not None:
            if isinstance(columns, list) or isinstance(columns, tuple):
                self.columns = columns
            else:
                raise NameError("Invalid type {}".format(type(columns)))
        else:
            self.columns = columns

    def fit(self, X, y=None, **fit_params):
        """Gets the columns to make a replace missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self

        """
        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns

        _lista = [i for i in self.columns if i not in X.columns.tolist()]
        if len(_lista) > 0:
            raise NameError("The columns {} no exist in Dataframe".format(_lista))

        self.train_median = X[self.columns].median()

        return self

    def transform(self, X):
        """this transformer handles missing values.

        Parameters
        ----------
        X : {Dataframe}, shape = [n_samples, n_features]
            Dataframe of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transform : {Dataframe}, shape = [n_samples, n_features]
            A copy of the input Dataframe with the columns replaced.

        """
        if not hasattr(self, "train_median"):
            raise AttributeError("FillNaTransformer_median has not been fitted, yet.")

        if not isinstance(X, pd.core.frame.DataFrame):
            raise NameError("Invalid type {}".format(type(X)))

        X[self.columns] = X[self.columns].fillna(self.train_median)

        return X




    # def remplace_na_sklearn(self, modo, value=0):
    #     """
    #     Remplazo de Valores NaN:
        
    #     Modos:
    #     - mean: remplazo por la media de la columna
    #     - median: remplazo por la mediana de la columna
    #     - idmax: remplazo por el idmax de la columna
    #     """
    #     from sklearn.preprocessing import Imputer

    #     categorical_list, numerical_list = categorical_vs_numerical()
    #     falta = self.data[feature].isnull().sum()
    #     print("valores faltantes: " + str(falta))

    #     if falta>0:
    #         if modo=="mean":
    #             print("Métoodo de remplazo: ", modo)
    #             self.data[numerical_list] = Imputer(strategy='mean').fit_transform(self.data[numerical_list])
    #         elif modo=="median":
    #             print("Método de remplazo: ", modo)
    #             self.data[numerical_list] = Imputer(strategy='median').fit_transform(self.data[numerical_list])
    #         elif modo=="idmax":
    #             print("Método de remplazo: ", modo)
    #             self.data[numerical_list] = Imputer(strategy='most_frequent').fit_transform(self.data[categorical_list])
    #         else:
    #             print("No se ha seleccionado ningun modo de remplazo")
    #     else:
    #         print("No se requiere de ningún remplazo de valores nulos") 



    # def remplace_na(self, feature, modo, value=0):
    #     """
    #     Remplazo de Valores NaN:
        
    #     Modos:
    #     - mean: remplazo por la media de la columna
    #     - median: remplazo por la mediana de la columna
    #     - idmax: remplazo por el idmax de la columna
    #     - any: Borra fila que haya algun NaN (how='any')
    #     - all: Borra fila que haya todo NaN (how='all')
    #     - value: rellenar el Nan con el value
    #     - forward: mas cerca hacia adelante
    #     - backward: mas cerca hacia atras
        
    #     """
    #     falta = self.data[feature].isnull().sum()
    #     print("valores faltantes: " + str(falta))

    #     if falta>0:
    #         if modo=="mean":
    #             print("Métoodo de remplazo: ", modo)
    #             self.data[feature].fillna(self.data[feature].mean(), inplace=True)
    #         elif modo=="median":
    #             print("Método de remplazo: ", modo)
    #             self.data[feature].fillna(self.data[feature].median(), inplace=True)
    #         elif modo=="idmax":
    #             print("Método de remplazo: ", modo)
    #             self.data[feature].fillna(self.data[feature].value_counts().idxmax(), inplace=True)
    #         elif modo=="any":
    #             print("Método de remplazo: ", modo)
    #             self.data.dropna(axis=0, how="any", inplace = True)
    #         elif modo=="all":
    #             print("Método de remplazo: ", modo)
    #             self.data.dropna(axis=0, how="all", inplace = True)
    #         elif modo=="value":
    #             print("Método de remplazo: ", modo)
    #             self.data[feature].fillna(value, inplace=True)
    #         elif modo=="forward":
    #             print("Método de remplazo: ", modo)
    #             self.data[feature].fillna(method="ffill", inplace = True)
    #         elif modo=="backward":
    #             print("Método de remplazo: ", modo)
    #             self.data[feature].fillna(method="bfill", inplace = True)
    #         else:
    #             print("No se ha seleccionado ningun modo de remplazo")
    #     else:
    #         print("No se requiere de ningún remplazo de valores nulos") 
