"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import load_model


class Neural(BaseEstimator, TransformerMixin):
    def __init__(self, clf, random_state=99, name="Neural_model"):
        self.clf = clf
        self.random_state = random_state
        self.name = name

    @classmethod
    def restore(cls, filename, random_state=99, name="Neural_model"):
        clf = load_model(filename)
        return cls(clf, random_state, name)

    def save_model(self, path, name=None):
        if name is None:
            name_save = self.name + ".h5"
        else:
            name_save = name + ".h5"
        filename = os.path.join(path, name_save)
        self.clf.save(filename)

        return filename

    def summary(self):
        self.clf.summary()

    def fit(self, X, y, **params):
        _history = self.clf.fit(X, y, **params)
        return _history

    def fit_cv(self, X, y, n_splits=10, shuffle=True, random_state=99, mute=False, 
                display=True, **params):

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cvscores = []
        n_history = []
        models = []
        tempdir = "checkpoints/temp/"
        filename = self.save_model(tempdir)
        i = 0
        for train, test in kfold.split(X, y):
            # clone model
            _model = load_model(filename)
            # fit
            start = time.time()
            history = _model.fit(X[train], y[train], verbose=0, validation_data=(X[test], y[test]), **params)
            n_history.append(history)
            end = time.time() - start
            val_acc = np.mean(history.history["val_accuracy"])*100
            print("Fold %s: %.3f%%,  metric: %s, time: %.3f sec." % (i+1, val_acc,
                                                                        _model.metrics_names[1],
                                                                        end))
            cvscores.append(val_acc)
            models.append(_model)
            i = i+1
        if not mute:
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        if display:
            self.plot_history(n_history)
        os.remove(filename)

        return models

    def plot_history(self, n_history):
        """Gráfica Resultados."""
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        ax = axs.flatten()
        fig.suptitle("Train-Validation", fontsize=30)
        i = 0
        for history in n_history:

            loss = history.history["loss"]
            accuracy = history.history["accuracy"]
            val_loss = history.history["val_loss"]
            val_accuracy = history.history["val_accuracy"]

            name = "Fold " + str(i+1)

            ax[0].plot(loss, label=name)
            ax[0].set_title("Train-loss")
            ax[0].legend(loc="upper right")

            ax[1].plot(accuracy, label=name)
            ax[1].set_ylim(0.6, 1.1)
            ax[1].set_title("Train-accuracy")
            ax[1].legend(loc="lower right")

            ax[2].plot(val_loss, label=name)
            ax[2].set_title("Validation-loss")
            ax[2].legend(loc="upper right")

            ax[3].plot(val_accuracy, label=name)
            ax[3].set_ylim(0.6, 1.1)
            ax[3].set_title("Validation-accuracy")
            ax[3].legend(loc="lower right")

            i = i+1
        plt.show()

    def predict(self, X, y=None):
        _predictions = tf.argmax(self.clf.predict(X), axis=1, output_type=tf.int32)
        return _predictions.numpy()

    def predict_proba(self, X, y=None):
        return self.clf.predict(X)

    def score(self, X, y):
        _predictions = tf.argmax(self.predict(X), axis=1, output_type=tf.int32)
        _preds = _predictions.numpy()
        print("Accuracy score: %.3f%%" % (accuracy_score(y, _preds)))
        return accuracy_score(y, _preds)

    def line(self):
        print("="*60)
        print("\n")

    def train_test(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def cuarentena(self, X, y, IDs):
        _preds = self.predict(X)
        df = pd.DataFrame()
        df["Id_cat"] = IDs
        df["Result"] = _preds != y
        del _preds
        return df

    def _evaluacion_rf_2features(self, data_eval, data_eval_target, save=False, logdir_report="", display=True):
        """
        Funcion que nos selecciona el thresholder más optimo:

        Inputs:

            - data_eval: Conjunto de evaluacion.
            - probs: Matriz de probabilidades que viene dado por el "lm.predict_proba" del podelo de RL.
            - thresholders: Intervalo de corte para determinar la categoria. viene dado por el "metrics.roc_curve" de sckitlearn.
            - display: si queremos mostrar los datos graficados por pantalla.

        Outputs:

            - th: Thresholder seleccionado.
            - res: El accurancy con ese thresholder.
            - df: Dataframe con el resultado del test.
        """
        results = []

        probs = self.predict_proba(data_eval)
        prob = probs[:, 1]
        df = pd.DataFrame(prob)

        _, _, thresholders = roc_curve(data_eval_target, prob)

        for i in range(len(thresholders)):

            df["prediction"] = np.where(df[0] >= thresholders[i], 1, 0)
            df["actual"] = list(data_eval_target)
            df["result"] = np.where(df["prediction"] == df["actual"], 1, 0)

            resultado = df["result"].sum()/df["result"].count()
            results.append(resultado)

        index = results.index(max(results))
        th = thresholders[index]
        res = max(results)

        if display:

            plt.plot(thresholders, results, marker="o", linestyle="--", color="r")
            x = [th, th]
            y = [0, 1]
            plt.plot(x, y)
            plt.xlabel("thresholders")
            plt.ylabel("Result")
            plt.title("Resultado segun thresholders")
            plt.xlim(-0.1, 1.1)
            plt.show()
            if save:
                plt.savefig(logdir_report + "_eval.png")

        df["prediction"] = np.where(df[0] >= thresholders[index], 1, 0)
        df["actual"] = list(data_eval_target)
        df["result"] = np.where(df["prediction"] == df["actual"], 1, 0)

        return th, res, df

    def Evaluation_model(self, X_train, X_test, y_train, y_test, clases=[0, 1], save=True, n_splits=10, ROC=True,
                            path="checkpoints/", **params):
        self.clf.fit(X_train, y_train, verbose=0, **params)

        y_pred = self.clf.predict(X_test)

        self.line()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, self.predict(X_test)))
        print("\n")
        print("Classification Report:")
        print(classification_report(y_test, self.predict(X_test), target_names=clases))

        self.line()
        th, res, df = self._evaluacion_rf_2features(X_test, y_test)
        print("----> Thresholder óptimo: {:.3f}, result: {:.3f}%".format(th, res*100))

        self.save_general(path, X_train, y_train)

    def save_general(self, path, X_train, y_train):
        ## Save Model
        self.line()
        print("Save Model")
        if not os.path.exists(path):
            os.makedirs(path)
            print("** Path creado: ", path)

        _filename = self.save_model(path, self.name)
        print("----> Modelo guardado en ", _filename)

    def Pipeline_train(self, X, y, X_train, X_test, y_train, y_test, n_splits=10, clases=[0, 1], ROC=True, path="checkpoints/", **params):
        print("="*60)
        print("  Pipeline: ", self.name)
        self.line()

        # Validacion cruzada sin optimizar
        self.fit_cv(X, y, n_splits=n_splits, **params)

        # Evaluación de resultados
        self.Evaluation_model(X_train, X_test, y_train, y_test, clases=clases, n_splits=n_splits, ROC=ROC, path=path, **params)


class Neural_sklearn(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=99, name="Neural_model_sklearn"):
        self.random_state = random_state
        self.name = name

    def buid_Pipeline(self, fn_clf, **params):
        """
        Example:
        def fn_clf(optimizer=tf.keras.optimizers.Adam(1e-3),
                 kernel_initializer='glorot_uniform',
                 dropout=0.2):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(8,)))
            model.add(tf.keras.layers.Dense(16, activation="relu",kernel_initializer=kernel_initializer))
            model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))

            model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
            return model

        params:
        {
            nb_epoch:100, 
            batch_size:32, 
            verbose:0
            }
        """
        ## Pipeline
        self.pipe = Pipeline([
                ('NN_keras', tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=fn_clf,
                                                                            **params))
                ])

    @classmethod
    def restore_pipeline(cls, filename, random_state=99, name="Neural_model_sklearn"):
        cls.pipe = pickle.load(open(filename, 'rb'))
        return cls(random_state, name)

    def build_param_grid(self, param_grid):
        if isinstance(param_grid, dict):
            self.param_grid = param_grid
        else:
            raise TypeError("Invalid type {}".format(type(param_grid)))

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

    def save(self, path, name=None):
        if name is None:
            name_save = self.name
        else:
            name_save = name
        filename = "Pipeline_" + name_save + ".pkl"
        filename = os.path.join(path, filename)
        pickle.dump(self.pipe, open(filename, 'wb'))
        print("----> Pipeline guardado en ", filename)
        return filename

    def summary(self):
        clf = self.pipe.named_steps['model']
        clf.summary()

    def fit(self, X, y):
        _history = self.pipe.fit(X, y)
        return _history

    def fit_cv(self, X, y, n_splits=10, shuffle=True, scoring="accuracy",
                mute=False, display=True):

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        print("Result CV sin optimizar: {:.3f}%  --  Metric: {}".format(
                cross_val_score(self.pipe, X, y, cv=kfold, scoring=scoring).mean()*100, scoring))

    def predict(self, X, y=None, binary=False):
        if not binary:
            _predictions = tf.argmax(self.pipe.predict(X), axis=1, output_type=tf.int32).numpy()
        else:
            _predictions = self.pipe.predict(X).reshape(-1)
        return _predictions

    def predict_proba(self, X, y=None):
        return self.predict(X)

    def score(self, X, y, binary=True):
        _preds = self.predict(X)
        print("Accuracy score: %.3f%%" % (accuracy_score(y, _preds)))
        return accuracy_score(y, _preds)

    def line(self):
        print("="*60)
        print("\n")

    def train_test(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def cuarentena(self, X, y, IDs):
        _preds = self.predict(X)
        df = pd.DataFrame()
        df["Id_cat"] = IDs
        df["Result"] = _preds != y
        del _preds
        return df

    def _evaluacion_rf_2features(self, data_eval, data_eval_target, save=False, binary=True,
                                    logdir_report="", display=True):
        """
        Funcion que nos selecciona el thresholder más optimo:

        Inputs:

            - data_eval: Conjunto de evaluacion.
            - probs: Matriz de probabilidades que viene dado por el "lm.predict_proba" del podelo de RL.
            - thresholders: Intervalo de corte para determinar la categoria. viene dado por el "metrics.roc_curve" de sckitlearn.
            - display: si queremos mostrar los datos graficados por pantalla.

        Outputs:

            - th: Thresholder seleccionado.
            - res: El accurancy con ese thresholder.
            - df: Dataframe con el resultado del test.
        """
        results = []

        probs = self.predict_proba(data_eval)
        if not binary:
            prob = probs[:, 1]
        else:
            prob = probs
        df = pd.DataFrame(prob)

        _, _, thresholders = roc_curve(data_eval_target, prob)

        for i in range(len(thresholders)):

            df["prediction"] = np.where(df[0] >= thresholders[i], 1, 0)
            df["actual"] = list(data_eval_target)
            df["result"] = np.where(df["prediction"] == df["actual"], 1, 0)

            resultado = df["result"].sum()/df["result"].count()
            results.append(resultado)

        index = results.index(max(results))
        th = thresholders[index]
        res = max(results)

        if display:

            plt.plot(thresholders, results, marker="o", linestyle="--", color="r")
            x = [th, th]
            y = [0, 1]
            plt.plot(x, y)
            plt.xlabel("thresholders")
            plt.ylabel("Result")
            plt.title("Resultado segun thresholders")
            plt.xlim(-0.1, 1.1)
            plt.show()
            if save:
                plt.savefig(logdir_report + "_eval.png")

        df["prediction"] = np.where(df[0] >= thresholders[index], 1, 0)
        df["actual"] = list(data_eval_target)
        df["result"] = np.where(df["prediction"] == df["actual"], 1, 0)

        return th, res, df

    def Evaluation_model(self, X, y, clases=[0, 1], save=True, n_splits=10, ROC=True, binary=True,
                            path="checkpoints/", **params):
        X_train, X_test, y_train, y_test = self.train_test(X, y)

        self.pipe.fit(X_train, y_train, **params)
        y_pred = self.predict(X_test)

        self.line()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, self.predict(X_test)))
        print("\n")
        print("Classification Report:")
        print(classification_report(y_test, self.predict(X_test), target_names=clases))

        if not binary:
            self.line()
            th, res, df = self._evaluacion_rf_2features(X, y, binary=binary)
            print("----> Thresholder óptimo: {:.3f}".format(th))


        """
        SIN SOLUCIÓN PARA EL GUARDADO
        """
        ## Save Model
        # self.line()
        # print("Save Model")
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #     print("** Path creado: ", path)

        # filename = "Pipeline_" + self.name + ".pkl"
        # filename = os.path.join(path, filename)
        # if not hasattr(self, "best_estimador"):
        #     self.pipe.fit(X_train, y_train)
        #     pickle.dump(self.pipe, open(filename, 'wb'))
        #     print("----> Pipeline guardado en ", filename)

    def Pipeline_train(self, X, y, n_splits=10, Randomized=False, n_iter=20, threshold='median',
                        clases=[0, 1], ROC=True, path="checkpoints/", eval=True):
        print("="*60)
        print("  Pipeline: ", self.name)
        self.line()

        ## Train-test-split
        X_train, _, y_train, _ = self.train_test(X, y)

        # Optimización
        if hasattr(self, "param_grid"):
            _, _ = self.Grid_model(X_train, y_train, Randomized=Randomized, n_iter=n_iter)

        # Validacion cruzada sin optimizar
        self.fit_cv(X_train, y_train, n_splits=n_splits)

        # Evaluación de resultados
        if eval:
            self.Evaluation_model(X, y, clases=clases, n_splits=n_splits, ROC=ROC,
                                    path="checkpoints/")


