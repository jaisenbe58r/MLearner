
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

# from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score

import catboost
from catboost import CatBoostClassifier
from catboost import MetricVisualizer

import seaborn as sns
import hyperopt
from numpy.random import RandomState
import scipy

from mlearner.utils import ParamsManager

import warnings
warnings.filterwarnings("ignore")

param_file = "mlearner/classifier/config/models.json"


class modelCatBoost(object):

    def __init__(self, name="CBT", random_state=99, *args, **kwargs):

        self.name = name
        self.train_dir = "model_" + str(self.name) + "/"
        self.random_state = random_state

        self.manager_models = ParamsManager(param_file, key_read="Models")
        self.params = self.manager_models.get_params()["CatBoost"]
        self.params.update({
                    'train_dir': self.train_dir,
                    "random_state": self.random_state
                })

        self.model = CatBoostClassifier(**self.params)

    def dataset(self, X, y, categorical_columns_indices=None, test_size=0.2, *args, **kwargs):

        self.categorical_columns_indices = categorical_columns_indices
        self.X = X
        self.columns = list(X)

        self.y, self.cat_replace = self.replace_multiclass(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)

        self.train_data = catboost.Pool(data=self.X_train.values,
                                        label=self.y_train.values,
                                        cat_features=self.categorical_columns_indices)
        self.eval_data = catboost.Pool(data=self.X_test.values,
                                        label=self.y_test.values,
                                        cat_features=self.categorical_columns_indices)
        self.all_train_data = catboost.Pool(data=self.X.values,
                                            label=self.y.values,
                                            cat_features=self.categorical_columns_indices)

    def replace_multiclass(self, targets):

        _unic = targets.unique().tolist()
        _remp = np.arange(0, len(_unic)).tolist()
        return targets.replace(_unic, _remp), _unic

    def fit(self, X, y, use_best_model=True, plot=True, save_snapshot=False, verbose=0, *args, **kwargs):

        self.dataset(X, y)
        _params = self.model.get_params()

        if verbose:
            _verbose = 0
        else:
            _verbose = _params["verbose"]

        return self.model.fit(self.train_data,
                                verbose=_verbose,
                                eval_set=self.eval_data,
                                use_best_model=use_best_model,
                                plot=plot,
                                save_snapshot=save_snapshot,
                                **kwargs)

        _preds = self.model.predict(self.dvalid)
        preds_test = np.where(_preds > 0.5, 1, 0)
        score_test = accuracy_score(self.y_test, preds_test)

        _preds = self.model.predict(self.dtrain)
        preds_train = np.where(_preds > 0.5, 1, 0)
        score_train = accuracy_score(self.y_train, preds_train)

        if not verbose == 0:
            print("Accurancy para el conjunto de entrenamiento ---> {:.2f}%".format(score_train*100))
            print("Accurancy para el conjunto de validacion ------> {:.2f}%".format(score_test*100))

    def fit_cv(self, X, y, fold_count=4, shuffle=True, stratified=True, plot=True, verbose=100):

        self.dataset(X, y)

        _params = self.model.get_params()
        _params.update({'verbose': verbose})

        _scores = catboost.cv(pool=self.all_train_data,
                                params=_params,
                                fold_count=fold_count,
                                seed=self.random_state,
                                shuffle=shuffle,
                                verbose=verbose,
                                plot=plot)
        if not verbose == 0:
            print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
        np.max(_scores['test-Accuracy-mean']),
        _scores['test-Accuracy-std'][np.argmax(_scores['test-Accuracy-mean'])],
        np.argmax(_scores['test-Accuracy-mean'])))

        return _scores

    def copy(self, *args, **kwargs):
        returned_classifier = CatBoostClassifier()
        returned_classifier.catboost_classifier = self.model.copy()
        returned_classifier.columns = self.columns
        return returned_classifier

    def update_model(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)

    def save_model(self, direct="./checkpoints", name="catboost_model"):

        if not os.path.isdir(direct):
            try:
                os.mkdir(direct)
                print("Directorio creado: " + direct)
            except OSError as e:
                raise NameError("Error al crear el directorio")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = direct + "/" + name + "_" + current_time + ".dump"
        self.model.save_model(filename)
        print("Modelo guardado en la ruta: " + filename)

    def load_model(self, direct="./checkpoints", name="catboost_model"):

        if not os.path.isdir(direct):
            print("no existe el drectorio especificado")
        filename = direct + "/" + name + ".dump"
        self.model.load_model(filename)
        print("Modelo cargado de la ruta: " + filename)

    def predict(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return self.model.predict(_X_copy.values, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        _X_copy = X.loc[:, self.columns].copy()
        return self.model.predict_proba(_X_copy.values, *args, **kwargs)

    def add_cat_features(self, index_features):

        self.categorical_columns_indices = index_features
        print(self.categorical_columns_indices)

        self.train_data = catboost.Pool(data=self.X_train,
                                        label=self.y_train,
                                        cat_features=self.categorical_columns_indices)
        self.eval_data = catboost.Pool(data=self.X_test,
                                        label=self.y_test,
                                        cat_features=self.categorical_columns_indices)
        self.all_train_data = catboost.Pool(data=self.X,
                                            label=self.y,
                                            cat_features=self.categorical_columns_indices)

    def index_features(self, features):

        _index = []
        for i in features:
            _index.append(self.X.columns.get_loc(i))
        if _index == []:
            raise NameError("No coincide ninguna de las features introducidas")
        return _index

    def get_important_features(self, display=True):

        self.model.get_feature_importance(prettified=True)
        _feature_importance_df = self.model.get_feature_importance(prettified=True)

        if display:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Importances", y="Feature Id", data=_feature_importance_df)
            plt.title('CatBoost features importance:')

        return _feature_importance_df

    def Visualizer_Models(self, directs=None, visu_model=True):

        directorios = []
        if len(directs) < 0:
            if visu_model:
                directorios.append(self.train_dir)
            else:
                raise NameError("No se ha seleccionado ningun directorio")
        else:
            if visu_model:
                directorios.append(self.train_dir)
            for i in directs:
                directorios.append(i)
        print(directorios)
        widget = MetricVisualizer(directorios)
        widget.start()

    def hyperopt_objective(self, params):

        _model = CatBoostClassifier(
            l2_leaf_reg=int(params['l2_leaf_reg']),
            learning_rate=params['learning_rate'],
            bagging_temperature=params["bagging_temperature"],
            iterations=500,
            eval_metric='AUC',
            random_seed=99,
            verbose=False,
            loss_function='Logloss'
            )
        _cv_data = catboost.cv(
            self.all_train_data,
            _model.get_params()
        )
        best_accuracy = np.max(_cv_data['test-AUC-mean'])

        return 1-best_accuracy

    def FineTune_hyperopt(self, X, y, mute=False):

        self.dataset(X, y)

        params_space = {
            'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
            'bagging_temperature': hyperopt.hp.uniform("bagging_temperature", 0, 0.3)
            }
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            self.hyperopt_objective,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=2,
            trials=trials,
            rstate=RandomState(self.random_state)
        )
        if not mute:
            print("\nBest parameters:")
            print(best)
            print("\n")

        _parameters = self.params
        _parameters.update(best)

        _model = CatBoostClassifier(**_parameters)
        _cv_data = catboost.cv(self.all_train_data, _model.get_params())

        if not mute:
            print('\nPrecise validation accuracy score: {}'.format(np.max(_cv_data['test-Accuracy-mean'])))
        return best

    def FineTune_sklearn(self, X, y, mute=False, n_splits=10, n_iter=2):
        """
        https://www.kaggle.com/ksaaskil/pets-definitive-catboost-tuning
        """
        self.dataset(X, y)

        def build_search(modelo, param_distributions, cv=5, n_iter=10, verbose=1, random_state=99):
            """
            Builder function for RandomizedSearch.
            """
            QWS = make_scorer(cohen_kappa_score, weights='quadratic')
            return RandomizedSearchCV(modelo,
                                      param_distributions=param_distributions,
                                      cv=cv,
                                      return_train_score=True,
                                      refit='cohen_kappa_quadratic',
                                      n_iter=n_iter,
                                      n_jobs=None,
                                      scoring={
                                            'accuracy': make_scorer(accuracy_score),
                                            'cohen_kappa_quadratic': QWS
                                      },
                                      verbose=verbose,
                                      random_state=random_state)
        def pretty_cv_results(cv_results,
                              sort_by='rank_test_cohen_kappa_quadratic',
                              sort_ascending=True,
                              n_rows=30):
            """
            Return pretty Pandas dataframe from the `cv_results_` attribute of finished parameter search,
            ranking by test performance and only keeping the columns of interest.
            """
            df = pd.DataFrame(cv_results)
            cols_of_interest = [key for key in df.keys()
                                    if key.startswith('param_')
                                        or key.startswith("mean_train")
                                        or key.startswith("std_train")
                                        or key.startswith("mean_test")
                                        or key.startswith("std_test")
                                        or key.startswith('mean_fit_time')
                                        or key.startswith('rank')]
            return df.loc[:, cols_of_interest].sort_values(by=sort_by, ascending=sort_ascending).head(n_rows)

        def run_search(X_train, y_train, search, mute=False):
            search.fit(X_train, y_train)
            print('Best score is:', search.best_score_)
            return pretty_cv_results(search.cv_results_)

        param_distributions = {

            'iterations': [100, 200],
            'learning_rate': scipy.stats.uniform(0.01, 0.3),
            'max_depth': scipy.stats.randint(3, 10),
            'one_hot_max_size': [30],
            'l2_leaf_reg': scipy.stats.reciprocal(a=1e-2, b=1e1),
        }

        if mute:
            _verbose = 0
        else:
            _verbose = 1

        self.params.update({
                            'use_best_model': False
                            })
        _model = CatBoostClassifier(**self.params)

        catboost_search = build_search(_model,
                                        param_distributions=param_distributions,
                                        n_iter=n_iter,
                                        verbose=_verbose,
                                        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=self.random_state))
        catboost_cv_results = run_search(self.X, self.y,
                                         search=catboost_search,
                                         mute=mute)
        best_estimator = catboost_search.best_estimator_
        if not mute:
            print(best_estimator.get_params())

        return catboost_cv_results, best_estimator

    def __getattr__(self, attr):
        """
        Pass all other method calls to self.model.
        """
        return getattr(self.model, attr)

"""
POR IMPLEMENTAR E INTEGRAR .....

#     def explainers_Shap(self):
#         
#         SHAP (SHapley Additive exPlanations) is a game theoretic approach
#         to explain the output of any machine learning model. It connects
#         optimal credit allocation with local explanations using the
#         classic Shapley values from game theory and their related extensions
#         
#         _cbc = CatBoostClassifier(**self.params)
        
#         _cbc.fit(self.train_data, # instead of X_train, y_train
#                       eval_set=self.eval_data, # instead of (X_valid, y_valid)
#                       use_best_model=True,
#                       plot=False)
        
#         _explainer = shap.TreeExplainer(_cbc) # insert your model
#         shap_values = _explainer.shap_values(self.train_data) # insert your train Pool object
        
        # Los primeros 100 valores
#         shap.initjs()
#         shap.force_plot(_explainer.expected_value, shap_values[:100,:], self.X_train.iloc[:100,:])
#         shap.summary_plot(shap_values, self.X_train)

"""
