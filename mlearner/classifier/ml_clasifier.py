
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from mlearner.training import Training
from mlearner.models import modelCatBoost, modelLightBoost, modelXGBoost
from mlearner.utils import ParamsManager
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/classifier/config/models.json"


class TrainingUtilities(object):
    def __init__(self, random_state=99):
        """
        Inicializacion de la clase de Preprocesado de un dataframe.
        """
        self.random_state = random_state
        self.manager_models = ParamsManager(param_file, key_read="Models")

    def Seleccion_rasgos(self, model, X, Y, n, display=True):
        """
        Funcion que nos retorna las categorias que mas influyen en el modelo.

        Inputs:
            - X, Y: dataset.
            - n: numero de categorias que queremos que formen parte de variables del modelo de RL.
            - display: Mostrar los resultados por pantalla

        Outputs:
            - z: Resumen de la inferencia de cada categoria en el modelo de RL.
            - valid: Categorias seleccionadas para formar parte del modelo de RL.
        """
        from sklearn.feature_selection import RFE

        categorias = list(X.columns)

        rfe = RFE(model, n)
        rfe = rfe.fit(X, Y.values.ravel())

        z = zip(X, rfe.support_, rfe.ranking_)
        categorias = list(X.columns)
        valid = []

        for i in range(len(rfe.support_)):

            if rfe.support_[i]:
                valid.append(categorias[i])

        if display:
            print("Resumen de las iteraciones para todas las categorias: \n")
            print(list(z))
            print("\n")
            print("Categorias seleccionadas: \n")
            print(valid)

        return z, valid

    def optimizacion_seleccion_rasgos(self, model, X, Y, n_splits, display=True):
        """
        Seleccion del numero de categorias a tener en cuenta
        """
        scores = []
        nums = []
        for i in range(len(X.columns)):

            _, valid = self.Seleccion_rasgos(model, X, Y, i+1)
            sc = self.validacion_cruzada(model, X[valid], Y, n_splits, shuffle=True)
            scores.append(sc)
            nums.append(i+1)

        d = {'n': nums, 'score': scores}
        df = pd.DataFrame(data=d)
        if display:
            plt.plot(df.n, df.score, marker="o", linestyle="--", color="r")
            plt.xlabel("numero categorias")
            plt.ylabel("Score")
            plt.grid(True)
            plt.title("Score frente al numero de categorias")

        return df

    def validacion_cruzada(self, model, X, Y, n_splits, shuffle=True, scoring="accuracy"):
        """
        Validacion cruzada del dataset introducido como input.

        Inputs:
            - cv = Numero de iteraciones.

        Outputs:
            - score: Media de los acc de todas las iteraciones.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, Y, scoring=scoring, cv=cv)

        return scores.mean()

    def opt_RandomForest_Classifier(self, X, Y, nmin=1, nmax=10000, num=4, display=True):
        """
        Seleccion del numero de categorias a tener en cuenta para RF
        """
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        rango = np.geomspace(nmin, nmax, num=num, endpoint=True)
        scores = []
        scores_val = []
        nums = []
        forestal = []

        for i in list(rango):

            forest = RandomForestClassifier(X_train, y_train, int(i))
            scores.append(forest.oob_score_)
            nums.append(int(i))
            forestal.append(forest)
            scores_val.append(forest.score(X_test, y_test))

        d = {'n': nums, 'score': scores, 'score_val': scores_val}
        df = pd.DataFrame(data=d)

        if display:

            plt.semilogx(df.n, df.score, marker="o", linestyle="--", color="r")
            plt.semilogx(df.n, df.score_val, marker="o", linestyle="--", color="b")
            plt.xlabel("numero categorias")
            plt.ylabel("Score")
            plt.grid(True)
            plt.title("Score frente al numero de categorias")

        return df, forestal


class PipelineClasificators(Training):

    def __init__(self, random_state=99):
        """
        Inicializacion de la clase de modelos
        """
        self.random_state = random_state
        self.manager_models = ParamsManager(param_file, key_read="Models")
        self.manager_finetune = ParamsManager(param_file, key_read="FineTune")

    def add_model(self, model):
        self.model = model

    def KNearestNeighbors(self):
        """
        """
        self.KNN = KNeighborsClassifier()
        return self.KNN

    def NaiveBayes(self):
        """
        Naive Bayes assumes the data to be normally distributed which can be
        achieved by scaling using the MaxAbsScaler.
        """
        self.NB = GaussianNB()
        return self.NB

    def RandomForestClassifier(self):
        """
        n_jobs: Parelizacion en la computacion.
        oob_score: True, muestreo aleatorio.
        n_estimadores = numero de arboles en el bosque
        max_features = numero maximo de caracteristicas consideradas para dividir un nodo
        max_depth = numero maximo de niveles en cada arbol de decision
        min_samples_split = numero minimo de puntos de datos colocados en un nodo antes de que el nodo se divida
        min_samples_leaf = numero minimo de puntos de datos permitidos en un nodo hoja
        bootstrap = metodo para muestrear puntos de datos (con o sin reemplazo)

        """
        self.modelRF = RandomForestClassifier(
            n_estimators=self.manager_models.get_params()["RandomForestClassifier"]["n_estimators"],
            criterion=self.manager_models.get_params()["RandomForestClassifier"]["criterion"],
            max_depth=self.manager_models.get_params()["RandomForestClassifier"]["max_depth"],
            min_samples_split=self.manager_models.get_params()["RandomForestClassifier"]["min_samples_split"],
            min_samples_leaf=self.manager_models.get_params()["RandomForestClassifier"]["min_samples_leaf"],
            min_weight_fraction_leaf=self.manager_models.get_params()["RandomForestClassifier"]["min_weight_fraction_leaf"],
            max_features=self.manager_models.get_params()["RandomForestClassifier"]["max_features"],
            min_impurity_decrease=self.manager_models.get_params()["RandomForestClassifier"]["min_impurity_decrease"],
            bootstrap=self.manager_models.get_params()["RandomForestClassifier"]["bootstrap"],
            oob_score=self.manager_models.get_params()["RandomForestClassifier"]["oob_score"],
            n_jobs=self.manager_models.get_params()["RandomForestClassifier"]["n_jobs"],
            random_state=self.random_state,
            verbose=self.manager_models.get_params()["RandomForestClassifier"]["verbose"],
            warm_start=self.manager_models.get_params()["RandomForestClassifier"]["warm_start"],
            # ccp_alpha=self.manager_models.get_params()["RandomForestClassifier"]["ccp_alpha"]
        )
        return self.modelRF

    def AdaBoostClassifier(self, **params):
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=self.random_state, **params)

    def GradientBoostingClassifier(self, **params):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=self.random_state, **params)

    def ExtraTreesClassifier(self, **params):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=self.random_state, **params)

    def SupportVectorMachine(self, **params):
        """
        """
        self.SVM = SVC(**params)
        return self.SVM

    def XGBoost(self, name="CBT"):
        """
        "min_child_weight": [ Minimum sum of instance weight (hessian) needed in a child.
        "objective": learning task.
        "eval_metric": Evaluation metrics for validation dat.
        "max_depth": Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
        "max_delta_step": /Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint.
        "sampling_method": The method to use to sample the training instances.
        "subsample": Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting.
        "eta": tep size shrinkage used in update to prevents overfitting.
        "gamma": Minimum loss reduction required to make a further partition on a leaf node of the tree.
        "lambda": L2 regularization term on weights. Increasing this value will make model more conservative.
        "alpha": L1 regularization term on weights. Increasing this value will make model more conservative.
        "tree_method":  he tree construction algorithm used in XGBoost.
        "predictor": The type of predictor algorithm to use.
        "num_parallel_tree": umber of parallel trees constructed during each iteration.

        https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        """
        self.modelXGBoost = modelXGBoost(name=name, random_state=self.random_state)
        return self.modelXGBoost

    def LightBoost(self, name="LBT"):

        self.LBoost = modelLightBoost(name=name, random_state=self.random_state)
        return self.LBoost

    def CatBoost(self, name="CBT"):

        self.CBoost = modelCatBoost(name=name, random_state=self.random_state)
        return self.CBoost

    def append_summary(self, model, X_train, X_test, y_train, y_test, name):
        train_start = time.perf_counter()
        score, _, _ = self.eval_Kfold_CV(model, X_train, X_test, y_train, y_test,
                                            n_splits=self.n_splits, shuffle=True, mute=True)
        train_end = time.perf_counter()
        prediction_start = time.perf_counter()
        _ = model.predict(X_test)
        prediction_end = time.perf_counter()

        self.names.append(name)
        self.utrain.append(train_end-train_start)
        self.utimes.append(prediction_end-prediction_start)

        return score

    def Pipeline_SelectModel(self, X, y, n_splits=5, select="XGBoost"):

        # Lista de modelos a optimizar
        self.scores = []
        self.names = []
        self.utrain = []
        self.utimes = []
        self.n_splits = n_splits
        self.features = X.columns.tolist()

        # Conjunto de entrenamiento y de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # KNearestNeighbors
        if self.manager_models.get_params()["select_models"]["KNN"]:
            _model = self.KNearestNeighbors()
            score = self.append_summary(_model, X_train, X_test, y_train, y_test, "KNearestNeighbors")
            self.scores.append(score)
            if select == "KNearestNeighbors":
                self.add_model(_model)
            print("Modelo: KNearestNeighbors --> Mean Accuracy: {:.3f}%\n".format(score*100))

        # NaiveBayes
        if self.manager_models.get_params()["select_models"]["NaiveBayes"]:
            from sklearn.preprocessing import MaxAbsScaler
            _model = self.NaiveBayes()
            scaler_gnb = MaxAbsScaler()
            sdss_train = scaler_gnb.fit_transform(X_train)
            sdss_test = scaler_gnb.fit_transform(X_test)
            pd_sdss_train = pd.DataFrame(columns=X_train.columns.tolist(), data=sdss_train)
            pd_sdss_test = pd.DataFrame(columns=X_test.columns.tolist(), data=sdss_test)
            score = self.append_summary(_model, pd_sdss_train, pd_sdss_test, y_train, y_test, "NaiveBayes")
            self.scores.append(score)
            if select == "NaiveBayes":
                self.add_model(_model)
            print("Modelo: NaiveBayes --> Mean Accuracy: {:.3f}%\n".format(score*100))

        # SupportVectorMachine
        if self.manager_models.get_params()["select_models"]["SVM"]:
            _model = self.SupportVectorMachine()
            score = self.append_summary(_model, X_train, X_test, y_train, y_test, "SupportVectorMachine")
            self.scores.append(score)
            if select == "SupportVectorMachine":
                self.add_model(_model)

            print("Modelo: SupportVectorMachine --> Mean Accuracy: {:.3f}%\n".format(score*100))

        # RandomForestClassifier
        if self.manager_models.get_params()["select_models"]["RandomForestClassifier"]:
            _model = self.RandomForestClassifier()
            score = self.append_summary(_model, X_train, X_test, y_train, y_test, "RandomForestClassifier")
            self.scores.append(score)
            if select == "RandomForestClassifier":
                self.add_model(_model)
            print("Modelo: RandomForestClassifier --> Mean Accuracy: {:.3f}%\n".format(score*100))

        # XGBoost
        if self.manager_models.get_params()["select_models"]["XGBoost"]:
            _model = self.XGBoost(name="XGBoost")
            _model.fit(X, y, verbose=0, mute=True)
            # score = self.append_summary(_model, X_train, X_test, y_train, y_test, "XGBoost")
            train_start = time.perf_counter()
            score, _ = _model.fit_cv(X, y, nfold=n_splits, verbose=0)
            self.scores.append(score)
            train_end = time.perf_counter()

            prediction_start = time.perf_counter()
            _ = _model.predict(X_test)
            prediction_end = time.perf_counter()

            if select == "XGBoost":
                self.add_model(_model)

            print("Modelo: XGBoost --> Mean Accuracy: {:.3f}%\n".format(score*100))

            self.names.append("XGBoost")
            self.utrain.append(train_end-train_start)
            self.utimes.append(prediction_end-prediction_start)

        # LightGBM
        if self.manager_models.get_params()["select_models"]["LightGBM"]:

            _model = self.LightBoost(name="LBT")
            _model.fit(X, y, verbose=0, mute=True)
            train_start = time.perf_counter()
            score, _ = _model.fit_cv(X, y, nfold=n_splits, verbose=0)
            self.scores.append(score)
            train_end = time.perf_counter()

            prediction_start = time.perf_counter()
            _ = _model.predict(X_test)
            prediction_end = time.perf_counter()

            if select == "LightGBM":
                self.add_model(_model.model)

            print("Modelo: LightGBM --> Mean Accuracy: {:.3f}%\n".format(score*100))

            self.names.append("LightGBM")
            self.utrain.append(train_end-train_start)
            self.utimes.append(prediction_end-prediction_start)

        # CatBoost
        if self.manager_models.get_params()["select_models"]["CatBoost"]:
            _model = self.CatBoost(name="CBT")
            train_start = time.perf_counter()
            score = _model.fit_cv(X, y, fold_count=self.n_splits, shuffle=True, stratified=True,
                                    plot=False, verbose=0)
            self.scores.append(np.mean(score["test-Accuracy-mean"]))
            train_end = time.perf_counter()

            _model.fit(X, y, plot=False, verbose=0)
            prediction_start = time.perf_counter()
            _model.model.predict(_model.eval_data)
            prediction_end = time.perf_counter()

            if select == "CatBoost":
                self.add_model(_model.model)

            print("Modelo: CatBoost --> Mean Accuracy: {:.3f}%\n".format(np.mean(score["test-Accuracy-mean"])*100))

            self.names.append("CatBoost")
            self.utrain.append(train_end-train_start)
            self.utimes.append(prediction_end-prediction_start)

        resultados = pd.DataFrame({"Modelo": self.names, "Mean Accuracy": self.scores,
                                    "Tiempo Entrenamiento": self.utrain, "Tiempo Prediccion": self.utimes})

        return resultados

    def Pipeline_SelectEmsembleModel(self, X, y, n_splits=10, mute=False, scoring="accuracy",
                                        display=True, save_image=False, path="/", AB=True):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        X_train, y_train = X, y
        ensembles = []
        if AB:
            ensembles.append(('AB', self.AdaBoostClassifier()))
        ensembles.append(('GBM', self.GradientBoostingClassifier()))
        ensembles.append(('ET', self.ExtraTreesClassifier()))
        ensembles.append(('RF', RandomForestClassifier(random_state=self.random_state)))
        ensembles.append(('XGB', XGBClassifier(random_state=self.random_state)))
        ensembles.append(('LGBM', LGBMClassifier(random_state=self.random_state)))

        results = []
        names = []
        for name, model in ensembles:
            kfold = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            # resultados = zip(name, results)
            if not mute:
                print(msg)

        scores = pd.DataFrame(np.asarray(results).T, columns=names)

        if display:
            figure, axs = plt.subplots(1, 2, figsize=(16, 5))
            ax = axs.flatten()

            # Compare Algorithms
            ax[0].set_title('Ensemble Algorithm Comparison')
            ax[0].boxplot(results)
            ax[0].set_xticklabels(names)
            if AB:
                axis = ["AB", "GBM", "ET", "RF", "XGB", "LGM"]
            else:
                axis = ["GBM", "ET", "RF", "XGB", "LGM"]
            scores_mean = np.mean(scores, axis=0)
            scores_std = np.std(scores, axis=0)
            ax[1].grid()
            ax[1].fill_between(axis, scores_mean - scores_std,
                                scores_mean + scores_std, alpha=0.1,
                                color="r")
            ax[1].plot(axis, scores_mean, 'o-', color="r",
                         label="CV score")
            ax[1].legend(loc="best")
            ax[1].set_title('Cross-validation score')
            figure.tight_layout()
            plt.show()

            if save_image:
                plt.savefig(path)

        return scores

    def Pipeline_FeatureSelect(self, X, y, n_splits=10, mute=False, scoring="accuracy", n_features=20,
                                display=True, save_image=False, path="/"):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        X_train, y_train = X, y
        models = []
        models.append(('GBM', self.GradientBoostingClassifier()))
        models.append(('ET', self.ExtraTreesClassifier()))
        models.append(('RF', RandomForestClassifier(random_state=self.random_state)))
        models.append(('XGB', XGBClassifier(random_state=self.random_state)))
        models.append(('LGBM', LGBMClassifier(random_state=self.random_state)))

        results = []
        names = []
        df = pd.DataFrame()

        for name, model in models:
            if not mute:
                print("modelo: {}".format(name))
            if not mute:
                print(".... Fitting")
            model.fit(X_train, y_train)
            if not mute:
                print(".... Permutation importance")
            result = permutation_importance(model, X_train, y_train, n_repeats=10,
                                            random_state=99)
            tree_importance_sorted_idx = np.argsort(model.feature_importances_)
            _ = np.arange(0, len(model.feature_importances_)) + 0.5

            name_features = "features_" + str(name)
            imp_features = "importance" + str(name)
            df[name_features] = X.columns[tree_importance_sorted_idx]
            df[imp_features] = model.feature_importances_[tree_importance_sorted_idx]

            features = df[name_features].values.tolist()[-n_features:]
            _X_train = X_train[features]
            _y_train = y_train
            if not mute:
                print(".... Select Features:")
                print(features)

            if not mute:
                print(".... Cross Validation")
            kfold = StratifiedKFold(n_splits=n_splits, random_state=99)
            cv_results = cross_val_score(model, _X_train, _y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            # resultados = zip(name, results)
            if not mute:
                print(".... Append Results:")
                print(msg)
                print("\n")

        scores = pd.DataFrame(np.asarray(results).T, columns=names)

        if display:
            figure, axs = plt.subplots(1, 2, figsize=(16, 5))
            ax = axs.flatten()

            # Compare Algorithms
            ax[0].set_title('Algorithm Comparison')
            ax[0].boxplot(results)
            ax[0].set_xticklabels(names)
            axis = ["GBM", "ET", "RF", "XGB", "LGM"]

            scores_mean = np.mean(scores, axis=0)
            scores_std = np.std(scores, axis=0)
            ax[1].grid()
            ax[1].fill_between(axis, scores_mean - scores_std,
                                scores_mean + scores_std, alpha=0.1,
                                color="r")
            ax[1].plot(axis, scores_mean, 'o-', color="r",
                         label="CV score")
            ax[1].legend(loc="best")
            ax[1].set_title('Cross-validation score')
            figure.tight_layout()
            plt.show()

            if save_image:
                plt.savefig(path)

        return scores, df

    def Pipeline_StackingClassifier(self, X, y, n_splits=5):
        # Lista de modelos
        self.models = []

        # KNearestNeighbors
        if self.manager_models.get_params()["stacking_models"]["KNN"]:
            _model = self.KNearestNeighbors()
            self.models.append(("KNearestNeighbors", _model))
        # NaiveBayes
        if self.manager_models.get_params()["stacking_models"]["NaiveBayes"]:
            _model = self.NaiveBayes()
            self.models.append(("NaiveBayes", _model))
        # SupportVectorMachine
        if self.manager_models.get_params()["stacking_models"]["SVM"]:
            _model = self.SupportVectorMachine()
            self.models.append(("SupportVectorMachine", _model))
        # RandomForestClassifier
        if self.manager_models.get_params()["stacking_models"]["RandomForestClassifier"]:
            _model = self.RandomForestClassifier()
            self.models.append(("RandomForestClassifier", _model))
        # XGBoost
        if self.manager_models.get_params()["stacking_models"]["XGBoost"]:
            _model = self.XGBoost(name="XGBoost")
            self.models.append(("XGBoost", _model))
        # LightGBM
        if self.manager_models.get_params()["stacking_models"]["LightGBM"]:
            _model = self.LightBoost(name="LBT")
            self.models.append(("LightGBM", _model))
        # CatBoost
        if self.manager_models.get_params()["stacking_models"]["CatBoost"]:
            _model = self.CatBoost(name="CBT")
            self.models.append(("CatBoost", _model))

    def _cv_results(self, X_train, Y_train, model, kfold, name, verbose=1):
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        if verbose > 0:
            print(msg)
        return cv_results

    def Ablacion_relativa(self, pipeline, X, y, n_splits=10, mute=False, std=True, scoring="accuracy",
                            display=True, save_image=False, path="/"):
        kfold = StratifiedKFold(n_splits=n_splits, random_state=99)

        models = []
        models.append(('AB', self.AdaBoostClassifier()))
        models.append(('GBM', self.GradientBoostingClassifier()))
        models.append(('RF', RandomForestClassifier(random_state=self.random_state)))
        models.append(('ET', self.ExtraTreesClassifier()))
        models.append(('LGM', LGBMClassifier(random_state=self.random_state)))
        models.append(('XGB', XGBClassifier(random_state=self.random_state)))
        models.append(('SVM', self.SupportVectorMachine()))
        models.append(('KNN', self.KNearestNeighbors()))

        scores_mean = []
        scores_std = []
        names_models = []

        for name_model, model in models:

            names = []
            results = []
            name = "Inicial"

            if not mute:
                print("\n", name_model)

            resu = self._cv_results(X, y, model, kfold, name)
            results.append(resu)
            names.append(name)

            for name, transf in pipeline:
                X_train = transf.fit_transform(X, y)
                Y_train = y
                resu = self._cv_results(X_train, Y_train, model, kfold, name)
                results.append(resu)
                names.append(name)

            scores = pd.DataFrame(np.asarray(results).T, columns=names)
            scores_mean.append(np.mean(scores, axis=0))
            scores_std.append(np.std(scores, axis=0))
            names_models.append(name_model)

        if display:
            fig, ax = plt.subplots(figsize=(14, 6))

            for i in range(len(scores_mean)):
                valor = scores_mean[i]-scores_mean[i].iloc[0]
                if std:
                    ax.fill_between(names, valor - scores_std[i],
                                    valor + scores_std[i], alpha=0.1)
                ax.plot(names, valor, 'o-', label=names_models[i], alpha=0.9)

            ax.plot(names, np.zeros(len(names)), 'ro--', label="cero", alpha=0.9)

            ax.grid()
            ax.legend(loc="best")
            ax.set_title('Mejoras relativas al modelo Inicial')
            fig.tight_layout()
            fig.show()

        if save_image:
            plt.savefig(path)

        return scores_mean, scores_std

    def features_importances(self, clf, X, y, display=True, save_image=False, path="/"):
        import seaborn as sns
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99, stratify=y)

        clf.fit(X_train, y_train)
        print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

        result = permutation_importance(clf, X_train, y_train, n_repeats=10,
                                        random_state=99)

        tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
        _ = np.arange(0, len(clf.feature_importances_)) + 0.5

        df = pd.DataFrame()
        df["feature"] = X.columns[tree_importance_sorted_idx]
        df["importance"] = clf.feature_importances_[tree_importance_sorted_idx]

        if display:
            _, _ = plt.subplots(figsize=(10, 30))
            sns.barplot(x="importance", y="feature",
                            data=df.sort_values(by="importance", ascending=False))
            plt.title('Features (avg over folds)')
            plt.show()

        if save_image:
            plt.savefig(path)

        return df

    def eval_Kfold_CV(self, model, X, X_test, y, y_test, n_splits=3, shuffle=True, mute=True):

        _Model = Training.add_model(model)

        resultados, score_general_test = _Model.KFold_CrossValidation(model, X, y, n_splits=n_splits,
                                                                        shuffle=shuffle, mute=mute)
        _predictions = model.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=_predictions)

        return score, resultados, score_general_test

    def func_acc(self, prob_pred, y_target):

        _y_pred = np.zeros(len(prob_pred))

        for i in range(0, len(prob_pred)):
            _y_pred[i] = int(np.argmax(prob_pred[i]))

        accuracy = accuracy_score(_y_pred, y_target)

        return accuracy

    def pred_binary(self, prob_pred, y_target, th=0.5):
        return accuracy_score(y_target, np.where(prob_pred > th, 1, 0))

    def replace_multiclass(self, targets):

        _unic = targets.unique().tolist()
        _remp = np.arange(0, len(_unic)).tolist()
        return targets.replace(_unic, _remp), _unic

    def Pipeline_GridSearch(self):
        pass


class wrapper_model(BaseEstimator, TransformerMixin):
    """Wrapper for Estimator."""
    def __init__(self, clf, pipeline_preprocess, random_state=99, name="model", select=False):
        self.clf = clf
        self.random_state = random_state
        self.name = name
        self.buid_Pipeline(pipeline_preprocess, select=select)

    def buid_Pipeline(self, pipeline_preprocess, threshold="median", select=True):
        ## Pipeline
        if select:
            self.pipe = Pipeline([
                ("pre-select", Pipeline([
                    ("preprocess", pipeline_preprocess),
                    ("Select features", SelectFromModel(self.clf, threshold=threshold)),
                ])),
                ("model", self.clf)
                ])
        else:
            self.pipe = Pipeline([
                ("pre-select", Pipeline([
                    ("preprocess", pipeline_preprocess),
                ])),
                ("model", self.clf)
                ])

    def build_param_grid(self, param_grid):
        if isinstance(param_grid, dict):
            self.param_grid = param_grid
        else:
            raise TypeError("Invalid type {}".format(type(param_grid)))

    def fit(self, X, y):
        self.pipe = self.pipe.fit(X, y)

    def fit_cv(self, X, y, n_splits=10, scoring="accuracy", shuffle=False):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state) 
        print("Result CV sin optimizar: {:.3f}%  --  Metric: {}".format(
                cross_val_score(self.pipe, X, y, cv=kfold, scoring=scoring).mean()*100, scoring))

    def Grid_model(self, X, y, n_splits=10, scoring="accuracy", Randomized=False, n_iter=20):

        if not hasattr(self, "param_grid"):
            raise NameError("Not buid Param")

        kfold = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
        if Randomized:
            search = RandomizedSearchCV(self.pipe, self.param_grid, n_jobs=-1, cv=kfold, n_iter=n_iter)
        else:
            search = GridSearchCV(self.pipe, self.param_grid, n_jobs=-1, cv=kfold)
        search.fit(X, y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print("  ", search.best_params_)
        self.best_estimador = search.best_estimator_
        self.pipe = search.best_estimator_
        return search.best_params_, search.best_estimator_

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def score(self, X, y):
        return self.pipe.score(X, y)

    def line(self):
        print("="*60)
        print("\n")

    def Restore_model(self, filename="checkpoints/model.pkl"):
        self.clf = pickle.load(open(filename, 'rb'))
        return self.clf

    def Restore_Pipeline(self, filename="checkpoints/Pipeline_model.pkl"):
        self.pipe = pickle.load(open(filename, 'rb'))
        return self.pipe

    @classmethod
    def restore_pipeline_v1(cls, filename, random_state=99, name="Pipeline_model"):
        cls.pipe = pickle.load(open(filename, 'rb'))
        return cls(random_state, name)

    def train_test(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def cuarentena(self, X, y):
        _preds = self.pipe.predict(X)
        df = X["Id_cat"]
        df["Result"] = _preds != y
        del _preds
        return df

    def Evaluation_model(self, X, y, clases=[0, 1], save=True, ROC=True, n_splits=10,
                            path="checkpoints/"):

        X_train, X_test, y_train, y_test = self.train_test(X, y)

        if not hasattr(self, "best_estimador"):
            clf = self.pipe.named_steps['model']
            transf = self.pipe.named_steps['pre-select']
            self.pipe.fit(X_train, y_train)
            y_pred = self.pipe.predict(X_test)
        else:
            clf = self.best_estimador.named_steps['model']
            transf = self.best_estimador.named_steps['pre-select']
            self.best_estimador.fit(X_train, y_train)
            y_pred = self.best_estimador.predict(X_test)

        eva = Training(clf, random_state=self.random_state)
        _X = pd.DataFrame(transf.fit_transform(X, y))

        self.line()
        print("Confusion Matrix:")
        print(eva.confusion_matrix(y_test, y_pred))
        _ = eva.class_report(y_test, y_pred, clases=clases)

        self.line()
        print("ROC: Cross Validation:")
        resultados, score_general_test = eva.KFold_CrossValidation(clf, _X, y, shuffle=True,
                                                                    n_splits=n_splits, ROC=ROC)
        self.line()
        if not hasattr(self, "best_estimador"):
            clf = self.pipe.named_steps['model']
        else:
            clf = self.best_estimador.named_steps['model']
        th, res, df = eva.evaluacion_rf_2features(clf, _X, y)
        print("----> Thresholder óptimo: {:.3f}, result: {:.3f}%".format(th, res*100))

        self.save_general(path, X_train, y_train)

    def save_general(self, path, X_train, y_train):
        ## Save Model
        self.line()
        print("Save Model")
        if not os.path.exists(path):
            os.makedirs(path)
            print("** Path creado: ", path)

        filename = "Pipeline_" + self.name + ".pkl"
        filename = os.path.join(path, filename)
        if not hasattr(self, "best_estimador"):
            self.pipe.fit(X_train, y_train)
            pickle.dump(self.pipe, open(filename, 'wb'))
            print("----> Pipeline guardado en ", filename)
        else:
            self.best_estimador.fit(X_train, y_train)
            pickle.dump(self.best_estimador, open(filename, 'wb'))
            print("----> Pipeline guardado en ", filename)
        self.line()

    def Pipeline_train(self, X, y, n_splits=10, Randomized=False, n_iter=20, threshold='median',
                        clases=[0, 1], ROC=True, path="checkpoints/", eval=True, report=False):
        print("="*60)
        print("  Pipeline: ", self.name)
        self.line()

        # Validacion cruzada sin optimizar
        self.fit_cv(X, y, n_splits=n_splits)

        # Optimización
        if hasattr(self, "param_grid"):
            best_params_, best_estimator_ = self.Grid_model(X, y, Randomized=Randomized, n_iter=20)
            best_params_txt = os.path.join(path, "best_params.txt")

            with open(best_params_txt, 'a') as file_txt:
                file_txt.write(str(self.name + ":"))
                file_txt.write("\n")
                file_txt.write(str(best_params_))
                file_txt.write("\n"*2)
                file_txt.close()

        # Evaluación de resultados
        if eval:
            self.Evaluation_model(X, y, clases=clases, n_splits=n_splits, ROC=ROC, path=path)


class wrapper_pipeline(BaseEstimator, TransformerMixin):
    """Wrapper for Estimator."""
    def __init__(self, filename, name="model", random_state=99):
        self.random_state = random_state
        self.name = name
        self.pipe = pickle.load(open(filename, 'rb'))

    def build_param_grid(self, param_grid):
        if isinstance(param_grid, dict):
            self.param_grid = param_grid
        else:
            raise TypeError("Invalid type {}".format(type(param_grid)))

    def fit(self, X, y):
        self.pipe = self.pipe.fit(X, y)

    def fit_cv(self, X, y, n_splits=10, scoring="accuracy", shuffle=False):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state) 
        print("Result CV sin optimizar: {:.3f}%  --  Metric: {}".format(
                cross_val_score(self.pipe, X, y, cv=kfold, scoring=scoring).mean()*100, scoring))

    def Grid_model(self, X, y, n_splits=10, scoring="accuracy", Randomized=False, n_iter=20):

        if not hasattr(self, "param_grid"):
            raise NameError("Not buid Param")

        kfold = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
        if Randomized:
            search = RandomizedSearchCV(self.pipe, self.param_grid, n_jobs=-1, cv=kfold, n_iter=n_iter)
        else:
            search = GridSearchCV(self.pipe, self.param_grid, n_jobs=-1, cv=kfold)
        search.fit(X, y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print("  ", search.best_params_)
        self.best_estimador = search.best_estimator_
        self.pipe = search.best_estimator_
        return search.best_params_, search.best_estimator_

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def score(self, X, y):
        return self.pipe.score(X, y)

    def line(self):
        print("="*60)
        print("\n")

    def Restore_model(self, filename="checkpoints/model.pkl"):
        self.clf = pickle.load(open(filename, 'rb'))
        return self.clf

    def Restore_Pipeline(self, filename="checkpoints/Pipeline_model.pkl"):
        self.pipe = pickle.load(open(filename, 'rb'))
        return self.pipe

    def train_test(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def cuarentena(self, X, y):
        _preds = self.pipe.predict(X)
        df = X["Id_cat"]
        df["Result"] = _preds != y
        del _preds
        return df

    def Evaluation_model(self, X, y, clases=[0, 1], save=True, ROC=True, n_splits=10,
                            path="checkpoints/"):

        X_train, X_test, y_train, y_test = self.train_test(X, y)

        if not hasattr(self, "best_estimador"):
            clf = self.pipe.named_steps['model']
            transf = self.pipe.named_steps['pre-select']
            self.pipe.fit(X_train, y_train)
            y_pred = self.pipe.predict(X_test)
        else:
            clf = self.best_estimador.named_steps['model']
            transf = self.best_estimador.named_steps['pre-select']
            self.best_estimador.fit(X_train, y_train)
            y_pred = self.best_estimador.predict(X_test)

        eva = Training(clf, random_state=self.random_state)
        _X = pd.DataFrame(transf.fit_transform(X, y))

        self.line()
        print("Confusion Matrix:")
        print(eva.confusion_matrix(y_test, y_pred))
        _ = eva.class_report(y_test, y_pred, clases=clases)

        self.line()
        print("ROC: Cross Validation:")
        resultados, score_general_test = eva.KFold_CrossValidation(clf, _X, y, shuffle=True,
                                                                    n_splits=n_splits, ROC=ROC)
        self.line()
        if not hasattr(self, "best_estimador"):
            clf = self.pipe.named_steps['model']
        else:
            clf = self.best_estimador.named_steps['model']
        th, res, df = eva.evaluacion_rf_2features(clf, _X, y)
        print("----> Thresholder óptimo: {:.3f}, result: {:.3f}%".format(th, res*100))

        ## Save Model
        self.save_general(path, X_train, y_train)

    def save_general(self, path, X_train, y_train):
        ## Save Model
        self.line()
        print("Save Model")
        if not os.path.exists(path):
            os.makedirs(path)
            print("** Path creado: ", path)

        filename = "Pipeline_" + self.name + ".pkl"
        filename = os.path.join(path, filename)
        if not hasattr(self, "best_estimador"):
            self.pipe.fit(X_train, y_train)
            pickle.dump(self.pipe, open(filename, 'wb'))
            print("----> Pipeline guardado en ", filename)
        else:
            self.best_estimador.fit(X_train, y_train)
            pickle.dump(self.best_estimador, open(filename, 'wb'))
            print("----> Pipeline guardado en ", filename)
        self.line()

    def Pipeline_train(self, X, y, n_splits=10, Randomized=False, n_iter=20, threshold='median',
                        clases=[0, 1], ROC=True, path="checkpoints/", eval=True):
        self.line()
        print("="*60)
        print("  Pipeline: ", self.name)
        self.line()

        # Validacion cruzada sin optimizar
        self.fit_cv(X, y, n_splits=n_splits)

        # Optimización
        if hasattr(self, "param_grid"):
            _, _ = self.Grid_model(X, y, Randomized=Randomized, n_iter=20)

        # Evaluación de resultados
        if eval:
            self.Evaluation_model(X, y, clases=clases, n_splits=n_splits, ROC=ROC, path=path)

