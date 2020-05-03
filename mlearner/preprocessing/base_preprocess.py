
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlearner.load import DataLoad


class DataExploratory(DataLoad):

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
        Inicializacion de la clase de Preprocesado de un dataframe
        """
        if isinstance(data, pd.core.frame.DataFrame):
            self.data = data
        else:
            raise TypeError("Invalid type {}".format(type(data)))

    def dtypes(self, X=None):
        """
        retorno del tipo de datos por columna
        """
        if X is None:
            return self.data.dtypes
        else:
            return X.dtypes

    def missing_values(self, X=None):
        """
        Numero de valores vacios en el dataframe.
        """
        if X is None:
            # Number of missing in each column
            missing = pd.DataFrame(self.data.isnull().sum()).rename(columns={0: 'total'})
        else:
            missing = pd.DataFrame(X.isnull().sum()).rename(columns={0: 'total'})

        _miss = missing.sort_values('total', ascending=False)
        return _miss

    def isNull(self):
        if not self.missing_values()["total"].values.sum() == 0:
            print("Cuidado que existen valores nulos")
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
        Deteccion de de categorias con type "object"
        """
        var = [i for i in self.view_features() if self.data[i].dtype == np.object]
        return var

    def not_type_object(self):
        """
        Deteccion de de categorias con type "object" 
        """
        var = [i for i in self.view_features() if not self.data[i].dtype == np.object]
        return var


class DataAnalyst(DataExploratory):

    """Class for Preprocessed object for data analysis.

    Attributes
    ----------
    data: pd.DataFrame of Dataset

    Examples
    --------
    For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/preprocessing/DataAnalyst/

    """
    def __init__(self, data):
        super().__init__(data)
        """
        Inicializacion de la clase de Analisis de datos
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

    def _check_parameters_features(self, features):

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

        return _features

    def _check_parameters_target(self, target):

        if target is not None:
            if isinstance(target, list) or isinstance(target, tuple):
                if len(target) == 0:
                    raise NameError("Empty category List")
                elif len(target) > 1:
                    raise NameError("Only one category column can be selected")
                else:
                    if target[0] not in self.data.columns.tolist():
                        raise NameError("Target '{}' not included in dataset columns".format(target))
                    else:
                        _target = target[0]
            else:
                raise TypeError("Invalid type {}".format(type(target)))
        else:
            raise NameError("Target can not be 'None'")

        return _target

    def _check_parameters_category(self, category, target):

        if category is not None:
            if isinstance(category, list) or isinstance(target, tuple):
                if len(category) == 0:
                    raise NameError("Empty category List")
                elif len(category) > 1:
                    raise NameError("Only one category can be selected")
                else:
                    if category[0] not in list(self.data[target].unique()):
                        raise NameError("Category '{}' not included in targets".format(category[0]))
                    else:
                        _category = category[0]
            else:
                raise TypeError("Invalid type {}".format(type(category)))
        else:
            raise NameError("category can not be 'None'")

        return _category

    def _check_path(self, path):
        if not os.path.isdir(path):
            raise NameError("Invalid path {}".format(path))

    def boxplot(self, features=None, target=None, display=False, save_image=False, path="/"):
        """
        Funcion que realiza un BoxPlot sobre la dispesion de cada categoria
        respecto a los grupos de target.

        Inputs:
            - data: Datos generales del dataset.
            - features: categorias a analizar.
        """
        _features = self._check_parameters_features(features)
        _target = self._check_parameters_target(target)

        if save_image:
            self._check_path(path)

        _vars = [i for i in _features if not i == _target]

        height, width = self._get_size_plot(features=_features)
        figure, axs = plt.subplots(height, width, figsize=(width*3, height*3))
        ax = axs.flatten()

        for i in range(len(_vars)):
            _cont = list(self.data[_target].unique())
            for j in _cont:
                data_train_group_j = self.data.groupby(_target).get_group(j)
                ax[i].boxplot(data_train_group_j[_vars[i]], positions=np.array([len(_cont)-1]))
                ax[i].set_title(_vars[i])
                ax[i].legend(list(self.data[_target].unique()))

        if display:
            figure.tight_layout()
            plt.show()

        if save_image:
            plt.savefig(path)

    def dispersion_categoria(self, features=None, target=None, density=True, display=False, save_image=False, path="/"):
        """
        Funcion que realiza un plot sobre la dispesion de cada categoria respecto a los grupos de target.

        Inputs:
            - data: Datos generales del dataset.
            - features: categorias a analizar.

        """
        _features = self._check_parameters_features(features)
        _target = self._check_parameters_target(target)

        if save_image:
            self._check_path(path)

        _vars = [i for i in _features if not i == _target]

        height, width = self._get_size_plot(features=_features)
        figure, axs = plt.subplots(height, width, figsize=(width*3, height*3))
        ax = axs.flatten()

        for i in range(len(_vars)):
            for j in list(self.data[_target].unique()):

                data_train_group_j = self.data.groupby(_target).get_group(j)
                ax[i].hist(data_train_group_j[_vars[i]], alpha=0.7, density=density)

            ax[i].set_title(_vars[i])
            ax[i].legend(list(self.data[_target].unique()))

        if display:
            figure.tight_layout()
            plt.show()

        if save_image:
            plt.savefig(path)

    def sns_jointplot(self, feature1, feature2, target=None, categoria1=None,
                        categoria2=None, display=True, save_image=False, path="/"):
        import seaborn as sns

        _feature1 = self._check_parameters_features(feature1)
        _feature2 = self._check_parameters_features(feature2)
        _target = self._check_parameters_target(target)

        _f = list(self.data[_target].unique())

        _categoria1 = self._check_parameters_category(categoria1, _target)

        elem = self.data.groupby(target).get_group(_categoria1)
        g = sns.jointplot(_feature1, _feature2, data=elem, kind="reg", truncate=False, color="m", height=7)

        if categoria2 is not None:
            _categoria2 = self._check_parameters_category(categoria2, _target)

            elem = self.data.groupby(target).get_group(_categoria2)
            g.x = elem[feature1]
            g.y = elem[feature2]
            g.plot_joint(plt.scatter, marker='x', c='b', s=50)

        if display:
            plt.show()

        if save_image:
            self._check_path(path)
            g.savefig(path)

    def sns_pairplot(self, features=None, target=None, display=True, save_image=False, path="/", palette="husl"):
        import seaborn as sns

        _features = self._check_parameters_features(features)
        _target = self._check_parameters_target(target)

        if _target not in _features:
            _features.append(_target)

        g = sns.pairplot(self.data[_features], hue=_target, palette=palette)

        if display:
            plt.show()

        if save_image:
            self._check_path(path)
            g.savefig(path)

    def distribution_targets(self, target=None, display=True, save_image=False, path="/", palette="Set2"):
        import seaborn as sns

        _target = self._check_parameters_target(target)

        total = self.data[_target].shape[0]
        ax = sns.countplot(x=_target, palette=palette, data=self.data)
        sns.set(font_scale=1.5)
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        _ = plt.gcf()
        # fig.set_size_inches(10,5)
        ax.set_ylim(top=total)

        for p in ax.patches:
            ax.annotate('{:.2f}%'.format(100*p.get_height()/total), (p.get_x()+0.3, p.get_height()+50))

        plt.title('Distribution of {} Targets'.format(total))
        plt.xlabel('Categories')
        plt.ylabel('Frequency [%]')

        if display:
            plt.show()

        if save_image:
            self._check_path(path)
            plt.savefig(path)

    def corr_matrix(self, features=None, display=True, save_image=False, path="/"):
        """
        matriz de covarianza:

        Un valor positivo para r indica una asociacion positiva
        Un valor negativo para r indica una asociacion negativa.

        Cuanto mas cerca estar de 1cuanto mas se acercan los puntos de datos a una linea recta,
        la asociacion lineal es mas fuerte. Cuanto mas cerca este r de 0, lo que debilita la asociacion lineal.

        """
        import seaborn as sns

        _features = self._check_parameters_features(features)
        X = self.data[_features]

        sns.set(style="white")
        corr = X.corr()
        # Set up the matplotlib figure
        _, _ = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        if display:
            plt.show()

        if save_image:
            self._check_path(path)
            plt.savefig(path)

    def Xy_dataset(self, target=None):
        """
        Separar datos del target en conjunto (X, y)
        """
        _target = self._check_parameters_target(target)
        y = _target
        x = [i for i in self.data.columns.tolist() if i not in y]

        self.X = self.data[x]
        self.y = self.data[y]
