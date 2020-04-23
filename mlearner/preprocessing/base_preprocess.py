
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class DataAnalyst(DataLoad):

    """Class for Preprocessed object for data analysis.

    Attributes
    ----------
    data: `pd.DataFrame` of Dataset

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/DataAnalyst/

    """
    def __init__(self, data):
        super().__init__(data)
        """
        Inicialización de la clase de Analisis de datos
        """
        if isinstance(data, pd.core.frame.DataFrame):
            self.data = data
        else:
            raise TypeError("Invalid type {}".format(type(data)))
        self.data = data

    def _get_size_plot(self, features, width=4):
        """Size of plot."""
        _n = len(features)
        if not _n <= width:
            width = 4
            nm = _n/width
        else:
            if _n > 1:
                width = _n
            else:
                width = 2
            nm = 0
        height = int(nm)+1

        return height, width

    def boxplot(self, features=None, target=None, display=False, save_image=False, path="/"):
        """
        Función que realiza un BoxPlot sobre la dispesión de cada categoria
        respecto a los grupos de target.

        Inputs:
            - data: Datos generales del dataset.
            - features: categorias a analizar.
        """
        """Init filter categories"""
        if features is not None:
            if isinstance(features, list) or isinstance(features, tuple):
                if len(features) == 0:
                    raise NameError("Empty category List")
                else:
                    for i in features:
                        if i not in self.data.columns.tolist():
                            raise NameError("Feature '{}' not included in dataset targets".format(i))
                        else:
                            for i in self.data[features].dtypes:
                                if i == "object":
                                    raise NameError("Type Object not permited")
                            else:
                                _features = features
            else:
                raise TypeError("Invalid type {}".format(type(features)))
        else:
            _features = self.data.select_dtypes(exclude=["object"]).columns.tolist()

        if target is not None:
            if isinstance(target, list) or isinstance(target, tuple):
                if len(target) == 0:
                    raise NameError("Empty category List")
                elif len(target) > 1:
                    raise NameError("Only one category column can be selected")
                else:
                    if target not in self.data.columns.tolist():
                        print(self.data.columns.tolist())
                        raise NameError("Target '{}' not included in dataset columns".format(target))
                    else:
                        _target = target[0]
            else:
                raise TypeError("Invalid type {}".format(type(target)))
        else:
            raise NameError("Target can not be 'None'")

        if save_image:
            if os.path.isdir(path):
                raise NameError("Invalid path {}".format(path))

        _vars = [i for i in _features if not i == _target]

        height, width = self._get_size_plot(features=_features)
        figure, axs = plt.subplots(height, width, figsize=(width*3, height*3))
        ax = axs.flatten()

        for i in range(len(_vars)):
            for j in list(self.data[_target].unique()):
                data_train_group_j = self.data.groupby(_target).get_group(j)
                ax[i].boxplot(data_train_group_j[_vars[i]], positions=np.array([j]))
                ax[i].set_title(_vars[i])
                ax[i].legend(list(self.data[_target].unique()))

        if display:
            figure.tight_layout()
            plt.show()

        if save_image:
            plt.savefig(path)

    def dispersion_categoria(self, features, target="target"):
        """
        Función que realiza un plot sobre la dispesión de cada categoria respecto a los grupos de target.

        Inputs:
            - data: Datos generales del dataset.
            - features: categorias a analizar.

        """
        _vars = [i for i in features if not i==target]

        if not len(_vars)<=4:
            ancho = 4
            nm = len(_vars)/ancho
        else:
            if len(_vars)>1:
                ancho = len(_vars)
            else:
                ancho = 2
            nm = 0
        alto = int(nm)+1

        figure, axs = plt.subplots(alto, ancho, figsize=(ancho*5, alto*5))
        ax = axs.flatten()

        for i in range(len(_vars)):
            
            for j in list(self.data[target].unique()):
                
                data_train_group_j = self.data.groupby(target).get_group(j)
                ax[i].hist(data_train_group_j[_vars[i]], alpha = 0.7, density=True)
            
            ax[i].set_title(_vars[i])
            ax[i].legend(list(self.data[target].unique()))

        figure.tight_layout()
        plt.show()

    def sns_pairplot(self, features, target="categoria", palette="husl"):
        import seaborn as sns

        if not target in features:
            features.append(target)
        g = sns.pairplot(self.data[features], hue=target, palette=palette)

    def sns_jointplot(self, categoria, feature1, 
                    feature2, target="categoria",
                    categoria2 = None,
                    save_image=False,
                    direct=""):

            import seaborn as sns

            f = list(self.data[target].unique())
            if categoria not in f:
                raise NameError('Categoria no incluida en la lista', f)
            else:

                elem = self.data.groupby(target).get_group(categoria)
                g = sns.jointplot(feature1, feature2, data=elem,
                        kind="reg", truncate=False,
                        color="m", height=7)
            
            if not categoria2 == None:
                
                if categoria2 not in f:
                    raise NameError('Categoria no incluida en la lista', f)
                else:
                    elem = self.data.groupby(target).get_group(categoria2)
                    g.x = elem[feature1]
                    g.y = elem[feature2]
                    g.plot_joint(plt.scatter, marker='x', c='b', s=50)

            if save_image:
                g.savefig(direct)

    def distribution_targets(self, targets, display=True):
        
        ax = sns.countplot(x = targets, palette="Set2")
        sns.set(font_scale=1.5)
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        fig = plt.gcf()
        # fig.set_size_inches(10,5)
        ax.set_ylim(top=len(targets))

        for p in ax.patches:
            ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+50))

        plt.title('Distribution of {} Targets'.format(len(targets)))
        plt.xlabel('Categorias')
        plt.ylabel('Frequency [%]')

        if display:
            plt.show()

    def corr_matrix(self, X):
        """
        matriz de covarianza:

        Un valor positivo para r indica una asociación positiva
        Un valor negativo para r indica una asociación negativa.

        Cuanto más cerca está r de 1cuanto más se acercan los puntos de datos a una línea recta, 
        la asociación lineal es más fuerte. Cuanto más cerca esté r de 0, lo que debilita la asociación lineal.

        """
        sns.set(style="white")

        corr = X.corr()
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.show()