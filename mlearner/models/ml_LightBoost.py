
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
import joblib
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import lightgbm as lgb
from lightgbm import LGBMClassifier

import seaborn as sns

from mlearner.training import Training
from mlearner.utils import ParamsManager

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/classifier/config/models.json"


class modelLightBoost(Training, BaseEstimator, ClassifierMixin):
    """
    Ejemplo multiclass:
    https://www.kaggle.com/nicapotato/multi-class-lgbm-cv-and-seed-diversification
    """
    def __init__(self, name="LGB", random_state=99, train_dir="", params=None, *args, **kwargs):

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

        self.model = LGBMClassifier(**self.params)
        super().__init__(self.model, random_state=self.random_state)

    def get_params_json(self):
        self.manager_models = ParamsManager(param_file, key_read="Models")
        self.params = self.manager_models.get_params()["LightBoost"]

        self.manager_finetune = ParamsManager(param_file, key_read="FineTune")
        self.params_finetune = self.manager_finetune.get_params()["LightBoost"]

    def dataset(self, X, y, categorical_columns_indices=None, test_size=0.2, *args, **kwarg):

        self.categorical_columns_indices = categorical_columns_indices
        self.X = X
        self.columns = list(X)

        self.y, self.cat_replace = self.replace_multiclass(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)

        self.dtrain = lgb.Dataset(self.X_train.values, label=self.y_train.values,
                                    feature_name=self.X_train.columns.tolist())
        self.dvalid = lgb.Dataset(self.X_test.values, label=self.y_test.values,
                                    feature_name=self.X_test.columns.tolist())
        self.all_train_data = lgb.Dataset(self.X.values, label=self.y.values,
                                            feature_name=self.X.columns.tolist())

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

        self.dtrain = lgb.Dataset(self.X_train.values, label=self.y_train.values,
                                    feature_name=self.X_train.columns.tolist())
        self.dvalid = lgb.Dataset(self.X_test.values, label=self.y_test.values,
                                    feature_name=self.X_test.columns.tolist())
        self.all_train_data = lgb.Dataset(self.X.values, label=self.y.values,
                                            feature_name=self.X.columns.tolist())

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

        self.params.update({'verbose': verbose})

        self.model = lgb.train(
                                self.params,
                                self.dtrain,
                                num_boost_round=num_boost_round,
                                verbose_eval=verbose,
                                **kwargs)

        preds_test = [np.argmax(line) for line in self.model.predict(self.X_test, num_iteration=self.model.best_iteration)]
        score_test = accuracy_score(self.y_test, preds_test)

        preds_train = [np.argmax(line) for line in self.model.predict(self.X_train, num_iteration=self.model.best_iteration)]
        score_train = accuracy_score(self.y_train, preds_train)

        if not mute:
            print("Accurancy para el conjunto de entrenamiento ---> {:.2f}%".format(score_train*100))
            print("Accurancy para el conjunto de validacion ------> {:.2f}%".format(score_test*100))

    def fit_cv(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, nfold=5,
                use_best_model=True, verbose=200, nosplit=False, early_stopping_rounds=150, num_boost_round=2000, **kwargs):

        if not nosplit:
            self.dataset(X, y)
        else:
            self.set_dataset_nosplit(X_train, X_test, y_train, y_test)

        self.params.update({'verbose': verbose})
        self.lgb_cv = lgb.cv(
                                params=self.params,
                                train_set=self.all_train_data,
                                num_boost_round=num_boost_round,
                                stratified=True,
                                nfold=nfold,
                                seed=self.random_state,
                                early_stopping_rounds=early_stopping_rounds,
                                **kwargs)
        loss = self.params["metric"]
        optimal_rounds = np.argmin(self.lgb_cv[str(loss) + '-mean'])
        best_cv_score = min(self.lgb_cv[str(loss) + '-mean'])

        if not verbose == 0:
            print("\nOptimal Round: {}\nOptimal Score: {:.3f} + stdv:{:.3f}".format(
                optimal_rounds, best_cv_score, self.lgb_cv[str(loss) + '-stdv'][optimal_rounds]))

        results = {"Rounds": optimal_rounds,
                                "Score": best_cv_score,
                                "STDV": self.lgb_cv[str(loss) + '-stdv'][optimal_rounds],
                                "LB": None,
                                "Parameters": self.params}
        score = np.mean(self.lgb_cv[str(loss) + '-mean'])

        return score, results

    def func_acc(self, prob_pred, y_target):

        _y_pred = np.zeros(len(prob_pred))

        for i in range(0, len(prob_pred)):
            _y_pred[i] = int(np.argmax(prob_pred[i]))

        accuracy = accuracy_score(_y_pred, y_target)
        return accuracy

    def pred_binary(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        preds = self.model.predict(_X_copy, *args, **kwargs)
        return np.where(preds > 0.5, 1, 0)

    def pred_multiclass(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return [np.argmax(line) for line in self.model.predict(_X_copy, num_iteration=self.model.best_iteration)]

    def update_model(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)

    def save_model(self, direct="./checkpoints", name="LGM_model", file_model=".txt"):

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

    def load_model(self, direct="./checkpoints/LGM_model.txt", file_model=".txt"):

        if not os.path.isdir(direct):
            print("no existe el drectorio especificado")

        if file_model == ".txt":
            self.model = LGBMClassifier(model_file=direct)
        elif file_model == ".pkl":
            self.model = joblib.load(direct)
        else:
            raise NameError("Type {} not permited".format(file_model))
        print("Modelo cargado de la ruta: " + direct)

    def predict(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return self.model.predict(_X_copy, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return self.model.predict_proba(_X_copy, *args, **kwargs)

    def index_features(self, features):

        _index = []
        for i in features:
            _index.append(self.X.columns.get_loc(i))
        if _index == []:
            raise NameError("No coincide ninguna de las features introducidas")
        return _index

    def get_important_features(self, display=True, max_num_features=20):

        if display:
            lgb.plot_importance(self.model, max_num_features=max_num_features, figsize=(6, 6), title='Feature importance (LightGBM)')
            plt.show()
        # return _feature_importance_df

    def FineTune_SearchCV(self, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, params=None, params_finetune=None, ROC=False,
                            randomized=True, cv=10, display_ROC=True, verbose=0, n_iter=10, replace_model=True, nosplit=False, finetune_dir=""):

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
        self.model = LGBMClassifier(**self.params)

        self._best_Parameters, self.results_df = self.FineTune(self.model, self.X_train, self.y_train, self.params_finetune,
                                                                cv=cv, randomized=True, n_iter=n_iter, verbose=1)
        self.params.update(**self._best_Parameters)
        self.fit(self.X_train, self.y_train)

        print("\n")
        score = accuracy_score(self.y_test, self.pred_multiclass(self.X_test))
        print("\n")
        print("Resultado del conjunto de test con los parametros optimos: {:.2f}%".format(score*100))
        print("\n")
        print("Report clasificacion con el conjunto de test: ")
        self.evaluate(self.model, self.X_test, self.y_test)
        print("\n")
        print("Validacion cruzada con todos los datos del dataset: ")
        print("\n")
        self.KFold_CrossValidation(LGBMClassifier(**self._best_Parameters), self.X, self.y, n_splits=cv, ROC=ROC, shuffle=True, mute=False,
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
        self.model = LGBMClassifier(**self.params)

        _rd = np.random.uniform(0, n_max, n_iter).astype(np.int32).tolist()
        params_finetuneseed = {"seed": _rd}
        del(_rd)

        self._best_Parameters, self.results_df = self.FineTune(self.model, self.X, self.y,
                                                                params_finetuneseed, randomized=False, cv=cv, n_iter=n_iter,
                                                                verbose=1, mute=True)
        print("All Model Runtime: %0.2f Minutes" % ((time.time() - allmodelstart)/60))

        print("Diversificacion de la semilla - mean AUC: {:.2f}% - std AUC: {:.5f}".format(
                    self.results_df['mean_test_AUC'].mean()*100, self.results_df['std_test_AUC'].mean()))

        print("Diversificacion de la semilla - mean Acc: {:.2f}% - std Acc: {:.5f}".format(
                    self.results_df['mean_test_Accuracy'].mean()*100, self.results_df['std_test_Accuracy'].mean()))
        return self._best_Parameters, self.results_df

    def SeedDiversification_fs(self, X, y, params, n_iter=10, mute=False, logdir_report="", display=True, save_image=True):

        allmodelstart = time.time()
        # Run Model with different Seeds
        all_feature_importance_df = pd.DataFrame()
        _y, _ = self.replace_multiclass(y)
        all_seeds = np.random.uniform(1, 1000, n_iter).astype(np.int32).tolist()

        for seeds_x in all_seeds:
            modelstart = time.time()
            print("Seed: ", seeds_x,)
            # Go Go Go
            params["seed"] = seeds_x
            model = lgb.train(
                params,
                lgb.Dataset(X.values, label=_y.values),
                verbose_eval=100)

            # Feature Importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = X.columns.tolist()
            fold_importance_df["importance"] = model.feature_importance()
            all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

            # Submit Model Individually
        #     seed_submit(model= lgb_final, seed= seeds_x, X_test)
            if not mute:
                print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
                print("#"*50)
            del model

        cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
        best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature",
                    data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        if display:
            plt.show()
        if save_image:
            filename = logdir_report + 'lgb_importances.png'
            plt.savefig(filename)
        print("All Model Runtime: %0.2f Minutes" % ((time.time() - allmodelstart)/60))

    def __getattr__(self, attr):
        """
        Pass all other method calls to self.model.
        """
        return getattr(self.model, attr)
