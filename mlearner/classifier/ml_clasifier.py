
"""Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from mlearner.training import Training
from mlearner.models import modelCatBoost, modelLightBoost, modelXGBoost
from mlearner.utils import ParamsManager

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/clasifier/config/models.json"


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

    def SupportVectorMachine(self):
        """
        """
        self.SVM = SVC()
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

            _model.fit(X, y, plot=False, mute=True)
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

    def Pipeline_StackingClassifier(self, X, y, n_splits=5, select="XGBoost"):

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

    def features_importances(self, display=True):
        importances = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        })
        importances = importances.sort_values(by='Importance', ascending=False)
        importances = importances.set_index('Feature')

        if display:
            importances.plot.bar()

        return importances

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
