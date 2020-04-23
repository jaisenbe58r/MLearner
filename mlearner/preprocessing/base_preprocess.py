
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
from mlearner.load import DataLoad


class DataCleaner(DataLoad):

    """Class to preprocessed object for data cleaning.

    Attributes
    ----------
    data: `pd.DataFrame` of Dataset

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/DataCleaner/

    """
    def __init__(self, data):
        super().__init__(data)
        """
        Inicialización de la clase de Preprocesado de un dataframe
        """
        if isinstance(data, pd.core.frame.DataFrame):
            self.data = data
        else:
            raise TypeError("Invalid type {}".format(type(data)))

    def dtypes(self):
        """
        retorno del tipo de datos por columna
        """
        return self.data.dtypes

    def missing_values(self):
        """
        Número de valores vacios en el dataframe.
        """
        # Number of missing in each column
        missing = pd.DataFrame(self.data.isnull().sum()).rename(columns={0: 'total'})
        _miss = missing.sort_values('total', ascending=False)
        return _miss

    def isNull(self):
        if not self.missing_values()["total"].values.sum() == 0:
            print("Cuidado que existen valore nulos")
            return True
        else:
            print("No existen valores nulos")
            return False

    def view_features(self):
        """
        Mostrar features del dataframe
        """
        return list(self.data.columns)

    def categorical_vs_numerical(self):
        categorical_list = []
        numerical_list = []
        for i in self.data.columns.tolist():
            if self.data[i].dtype == 'object':
                categorical_list.append(i)
            else:
                numerical_list.append(i)
        print('Number of categorical features:', str(len(categorical_list)))
        print('Number of numerical features:', str(len(numerical_list)))

        return categorical_list, numerical_list

    def type_object(self):
        """
        Detección de de categorias con type "object"
        """
        var = [i for i in self.view_features() if self.data[i].dtype == np.object]
        return var

    def not_type_object(self):
        """
        Detección de de categorias con type "object" 
        """
        var = [i for i in self.view_features() if not self.data[i].dtype == np.object]
        return var
