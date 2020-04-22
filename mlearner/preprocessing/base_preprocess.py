
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlearner.load import DataLoad


class Data_cleaner(DataLoad):

    def __init__(self, data): 
        super().__init__(data)
        """
        Inicialización e la clase de Preprocesado de un dataframe
        """
        self.data = data

    def dtypes(self):
        """
        retorno del tipo de datos por columna
        """
        return self.data.dtypes
    
    def missing_values(self):
        """
        Número de valores vacios en el dtaframe
        """
        # Number of missing in each column
        missing = pd.DataFrame(self.data.isnull().sum()).rename(columns = {0: 'total'})
        miss = missing.sort_values('total', ascending = False)
        return miss

    def isNull(self):
        if not self.missing_values()["total"].values.sum()==0:
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
   
    def remplace(self, features, mapping={"yes": 1, "no": 0}):
        """
        Remplazo de datos según el dicionario "mappping".
        """
        for f in features:
            # Fill in the values with the correct mapping
            self.data[f] = self.data[f].replace(mapping).astype(np.float64)

    def replace_multiclass(self, target="target"):
        
        self._unic = self.data[target].unique().tolist()
        _remp = np.arange(0, len(self._unic)).tolist()
        
        return self.data[target].replace(self._unic, _remp, inplace=True)
    
    def categorical_vs_numerical(self):
        categorical_list = []
        numerical_list = []
        for i in self.data.columns.tolist():
            if self.data[i].dtype=='object':
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
        var = [i for i in self.view_features() if self.data[i].dtype==np.object]
        return var
    
    def not_type_object(self):
        """
        Detección de de categorias con type "object" 
        """
        var = [i for i in self.view_features()  if not self.data[i].dtype==np.object]
        return var
    
    def extract_target(self, list_target=[0, 1], target="target", implace=False):

        if len(list_target) == 0:
            raise NameError("ERROR -- Lista de categoria vacia")
        if len(list_target) > len(self.data[target].unique().tolist()):
            raise NameError("ERROR -- Lista de categorias mayor que el numero de categorias")
        for i in list_target:
            if i not in self.data[target].unique().tolist():
                raise NameError("ERROR -- Categoria '{}' no incluida en los targets del dataset".format(i))

        _data_i0 = self.data[self.data[target]==list_target[0]]
        if len(list_target)>1:
            for i in range(len(list_target)-1):
                _data_i1 = self.data[self.data[target]==i+1]
                _data_i0 = pd.merge(left=_data_i0, right=_data_i1, how = "outer")

        if implace:
            self.data = _data_i0

        return _data_i0

