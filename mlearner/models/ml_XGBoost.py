
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier

import seaborn as sns

from mlearner.training import Training
from mlearner.utils import ParamsManager

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/classifier/config/models.json"


class modelXGBoost(Training, BaseEstimator, ClassifierMixin):
    """
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient,
    flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.
    XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science
    problems in a fast and accurate way. The same code runs on major distributed environment
    (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

    Parameters
    ----------
        "min_child_weight": [ Minimum sum of instance weight (hessian) needed in a child.
        "objective": learning task.
        "eval_metric": Evaluation metrics for validation data.
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
        ...

    Documentation
    -------------
        https://xgboost.readthedocs.io/en/latest/
        https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    """
    def __init__(self, name="XGB", random_state=99, train_dir="", params=None, *args, **kwargs):

        self.name = name
        self.train_dir = train_dir + "/" + "model_" + str(self.name) + "/"
        self.random_state = random_state

        if params is None:
            self.get_params_json()
            self.params.update({
                        'model_dir': self.train_dir,
                        "seed": self.random_state})
        else:
            # if isinstance(params)
            self.params = params

        self.model = XGBClassifier(**self.params)
        super().__init__(self.model, random_state=self.random_state)

    def get_params_json(self):
        self.manager_models = ParamsManager(param_file, key_read="Models")
        self.params = self.manager_models.get_params()["XGBoost"]

        self.manager_finetune = ParamsManager(param_file, key_read="FineTune")
        self.params_finetune = self.manager_finetune.get_params()["XGBoost"]

    def dataset(self, X, y, categorical_columns_indices=None, test_size=0.2, *args, **kwarg):

        self.categorical_columns_indices = categorical_columns_indices

        self.X = X
        self.columns = list(X)

        self.y, self.cat_replace = self.replace_multiclass(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dvalid = xgb.DMatrix(self.X_test, label=self.y_test)
        self.all_train_data = xgb.DMatrix(self.X, label=self.y)

    def set_dataset_nosplit(self, X_train, X_test, y_train, y_test, categorical_columns_indices=None, *args, **kwarg):

        self.categorical_columns_indices = categorical_columns_indices

        self.columns = list(X_train)

        _ytrain, _ = self.replace_multiclass(y_train)
        _ytest, _ = self.replace_multiclass(y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.X = pd.concat([X_train, X_test], axis=0)
        self.y = pd.concat([y_train, y_test], axis=0)

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dvalid = xgb.DMatrix(self.X_test, label=self.y_test)
        self.all_train_data = xgb.DMatrix(self.X, label=self.y)

    def replace_multiclass(self, targets):

        _unic = targets.unique().tolist()
        _remp = np.arange(0, len(_unic)).tolist()
        return targets.replace(_unic, _remp), _unic

    def fit(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, mute=False,
            use_best_model=True, verbose=0, num_boost_round=100, nosplit=False, **kwargs):

        if not nosplit:
            self.dataset(X, y)
        else:
            self.set_dataset_nosplit(X_train, X_test, y_train, y_test)

        self.params.update({'verbosity': verbose})

        self.model = xgb.train(
                            self.params,
                            self.dtrain,
                            num_boost_round=num_boost_round,
                            # verbosity=verbose,
                            **kwargs)

        _preds = self.model.predict(self.dvalid)
        preds_test = np.where(_preds > 0.5, 1, 0)
        score_test = accuracy_score(self.y_test, preds_test)

        _preds = self.model.predict(self.dtrain)
        preds_train = np.where(_preds > 0.5, 1, 0)
        score_train = accuracy_score(self.y_train, preds_train)

        if not mute:
            print("Accurancy para el conjunto de entrenamiento ---> {:.2f}%".format(score_train*100))
            print("Accurancy para el conjunto de validacion ------> {:.2f}%".format(score_test*100))

    def fit_cv(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, num_boost_round=75,
                nfold=5, use_best_model=True, verbose=2, nosplit=False, early_stopping_rounds=75, **kwargs):
        """
        https://xgboost.readthedocs.io/en/latest/parameter.html
        """
        if not nosplit:
            self.dataset(X, y)
        else:
            self.set_dataset_nosplit(X_train, X_test, y_train, y_test)

        self.params.update({'verbose_eval': verbose})
        self.xgb_cv = xgb.cv(
                            self.params,
                            self.all_train_data,
                            num_boost_round,
                            nfold,
                            early_stopping_rounds=early_stopping_rounds,
                            stratified=True,
                            seed=self.random_state)

        loss = "test-" + self.params["metrics"][0]
        optimal_rounds = np.argmin(self.xgb_cv[str(loss) + '-mean'])
        best_cv_score = max(self.xgb_cv[str(loss) + '-mean'])

        if not verbose == 0:
            print("\nOptimal Round: {}\nOptimal Score: {:.3f} + std:{:.3f}".format(
                optimal_rounds, best_cv_score, self.xgb_cv[str(loss) + '-std'][optimal_rounds]))

        results = {"Rounds": optimal_rounds,
                                "Score": best_cv_score,
                                "STDV": self.xgb_cv[str(loss) + '-std'][optimal_rounds],
                                "LB": None,
                                "Parameters": self.params}

        score = self.xgb_cv[str(loss) + '-mean'].mean()
        return score, results

    def func_acc(self, prob_pred, y_target):

        _y_pred = np.zeros(len(prob_pred))

        for i in range(0, len(prob_pred)):
            _y_pred[i] = int(np.argmax(prob_pred[i]))
        accuracy = accuracy_score(_y_pred, y_target)

        return accuracy

    def predict(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return self.model.predict(xgb.DMatrix(_X_copy), *args, **kwargs)

    def pred_binary(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        preds = self.model.predict(xgb.DMatrix(_X_copy), *args, **kwargs)
        return np.where(preds > 0.5, 1, 0)

    def pred_multiclass(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return [np.argmax(line) for line in self.model.predict(xgb.DMatrix(_X_copy))]

    def update_model(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)

    def save_model(self, direct="./checkpoints", name="XGB_model", file_model=".txt"):

        if not os.path.isdir(direct):
            try:
                os.mkdir(direct)
                print("Directorio creado: " + direct)
            except OSError as e:
                raise NameError("Error al crear el directorio")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if file_model == ".txt":
            filename = direct + "/" + name + "_" + current_time + ".txt"
            self.model.save_model(filename)
        elif file_model == ".pkl":
            filename = direct + "/" + name + "_" + current_time + ".pkl"
            joblib.dump(self.model, filename)
        else:
            raise NameError("Type {} not permited".format(file_model))
        print("Modelo guardado en la ruta: " + filename)

    def load_model(self, direct="./checkpoints/XGB_model.txt", file_model=".txt"):

        if not os.path.isdir(direct):
            print("no existe el drectorio especificado")

        if file_model == ".txt":
            self.model = XGBClassifier(model_file=direct)
        elif file_model == ".pkl":
            self.model = joblib.load(direct)
        else:
            raise NameError("Type {} not permited".format(file_model))
        print("Modelo cargado de la ruta: " + direct)

    def index_features(self, features):

        _index = []
        for i in features:
            _index.append(self.X.columns.get_loc(i))

        if _index == []:
            raise NameError("No coincide ninguna de las features introducidas")

        return _index

    def get_important_features(self, display=True, max_num_features=20):

        _model = XGBClassifier()
        _model.fit(self.X, self.y)
        _data = np.array([self.X.columns, _model.feature_importances_])
        _feature_importance_df = pd.DataFrame(_data.T, columns=["Feature Id", "Importances"])
        _feature_importance_df = _feature_importance_df.sort_values(by=['Importances'], ascending=False)

        if display:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Importances", y="Feature Id", data=_feature_importance_df)
            plt.title('XGBoost features importance:')
        # if display:
        #     xgb.plot_importance(self.model, max_num_features=max_num_features, figsize=(6, 6), title='Feature importance (LightGBM)')
        #     plt.show()

        return _feature_importance_df

    def FineTune_SearchCV(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, params=None, params_finetune=None, ROC=False,
                            randomized=True, cv=10, n_iter=10, replace_model=True, verbose=0, nosplit=False, finetune_dir=""):
        self.get_params_json()
        self.finetune_dir = finetune_dir + "/" + "model_finetune_" + str(self.name) + "/"
        self.params.update({
                    'train_dir': self.finetune_dir,
                    "seed": self.random_state})
        if params is not None:
            self.params = params
        if params_finetune is not None:
            self.params_finetune = params_finetune

        if not nosplit:
            self.dataset(X, y)
        else:
            self.set_dataset_nosplit(X_train, X_test, y_train, y_test)

        self.params.update({'verbosity': verbose})
        self.model = XGBClassifier(**self.params)

        self._best_Parameters, self.results_df = self.FineTune(self.model, self.X_train, self.y_train, self.params_finetune,
                                                                randomized=True, cv=cv, n_iter=n_iter, verbose=1)
        self.params.update(**self._best_Parameters)
        self.fit(self.X_train, self.y_train, verbose=1)

        score = accuracy_score(self.y_test, self.pred_multiclass(self.X_test))
        print("Resultado del conjunto de test con los parametros optimos: {:.2f}%".format(score*100))
        print("\n")
        print("Report clasificacion con el conjunto de test: ")
        self.evaluate(self.model, xgb.DMatrix(self.X_test), self.y_test)
        print("\n")
        print("Cross validation con todos los datos del dataset: ")
        print("\n")
        self.KFold_CrossValidation(XGBClassifier(**self._best_Parameters), self.X_test, self.y_test, n_splits=cv, ROC=ROC, shuffle=True, mute=False,
                                    logdir_report="", display=True, save_image=True, verbose=0)

        return self._best_Parameters, self.results_df

    def SeedDiversification_cv(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, n_iter=10, n_max=2018-2022, cv=10,
                                nosplit=False, finetuneseed_dir="", display=True, save_image=True, verbose=0):
        allmodelstart = time.time()
        self.get_params_json()

        self.finetune_dir = finetuneseed_dir + "/" + "model_finetune_seed" + str(self.name) + "/"
        self.params.update({'train_dir': self.finetune_dir,
                            'verbosity': verbose})
        if not nosplit:
            self.dataset(X, y)
        else:
            self.set_dataset_nosplit(X_train, X_test, y_train, y_test)

        self.params.update({'verbosity': verbose})
        self.model = XGBClassifier(**self.params)

        _rd = np.random.uniform(0, n_max, n_iter).astype(np.int32).tolist()
        params_finetuneseed = {"seed": _rd}
        del(_rd)

        self._best_Parameters, self.results_df = self.FineTune(self.model, self.X, self.y, params_finetuneseed,
                                                                randomized=False, cv=cv, n_iter=n_iter,
                                                                verbose=1, mute=True)

        print("All Model Runtime: %0.2f Minutes" % ((time.time() - allmodelstart)/60))

        print("Diversificacion de la semilla - mean AUC: {:.2f}% - std AUC: {:.5f}".format(
                    self.results_df['mean_test_AUC'].mean()*100, self.results_df['std_test_AUC'].mean()))

        print("Diversificacion de la semilla - mean Acc: {:.2f}% - std Acc: {:.5f}".format(
                    self.results_df['mean_test_Accuracy'].mean()*100, self.results_df['std_test_Accuracy'].mean()))

        return self._best_Parameters, self.results_df

    def __getattr__(self, attr):
        """
        Pass all other method calls to self.model.
        """
        return getattr(self.model, attr)
