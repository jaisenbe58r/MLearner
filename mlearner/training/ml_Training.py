
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
from sklearn.metrics import classification_report

from mlearner.evaluation import EvaluationModels

import warnings
warnings.filterwarnings("ignore")


class Training(EvaluationModels):
    def __init__(self, model, random_state=99):
        self.model = model
        self.random_state = random_state
        super().__init__(self.model, self.random_state)

    @classmethod
    def add_model(cls, model, random_state=99):
        """
        Incorporar  modelo en la clase
        """
        return cls(model, random_state)

    def get_model(self):
        return self.model

    def KFold_CrossValidation(self, model, X, y, n_splits=10, ROC=False, shuffle=True, mute=False,
                                logdir_report="", display=True, save_image=True, verbose=0):
        """
        Validacion cruzada respecto a "n_plits" del KFolds.
        """
        y, targets = self.replace_multiclass(y)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        results = pd.DataFrame(columns=['training_score', 'test_score'])
        fprs, tprs, scores = [], [], []

        if not ROC:
            for (train, test), i in zip(kf.split(X, y), range(n_splits)):
                model.fit(X.iloc[train], y.iloc[train])

                predictions_train = model.predict(X.iloc[train])
                predictions_test = model.predict(X.iloc[test])

                score_train = accuracy_score(y.iloc[train], predictions_train)
                score_test = accuracy_score(y.iloc[test], predictions_test)

                scores.append((score_train, score_test))
                if not mute:
                    print("  * Fold {0} accuracy: {1}".format(i+1, score_test))

            resultados = pd.DataFrame(scores, columns=['Score Train', 'Score Test'])
            score_general_test = resultados['Score Test'].mean()

        else:
            for (train, test), i in zip(kf.split(X, y), range(n_splits)):
                model.fit(X.iloc[train], y.iloc[train])
                _, _, auc_score_train = self.compute_roc_auc(model, train, X, y)
                fpr, tpr, auc_score = self.compute_roc_auc(model, test, X, y)

                predictions_train = model.predict(X.iloc[train])
                predictions_test = model.predict(X.iloc[test])

                score_train = accuracy_score(y.iloc[train], predictions_train)
                score_test = accuracy_score(y.iloc[test], predictions_test)

                scores.append((auc_score_train, auc_score, score_train, score_test))
                fprs.append(fpr)
                tprs.append(tpr)

            self.create_ROC_pro(fprs, tprs, X, y, targets=targets, logdir_report=logdir_report,
                                    display=display, save_image=save_image)
            resultados = pd.DataFrame(scores, columns=['AUC Train', 'AUC Test',
                                                        'Acc Train', 'Acc Test'])
            score_general_test = resultados['Acc Test'].mean()

        if not mute:
            print("\n")
            print("----> Resultado de la validacion cruzada: {:.3f}%".format(score_general_test*100))
            if ROC:
                print("----> Resultado de la validacion cruzada AUC: {:.3f}%".format(
                                                                resultados['AUC Test'].mean()*100))
        return resultados, score_general_test

    def FineTune(self, model, X, y, params, refit='Accuracy', cv=3, verbose=0, randomized=True, n_iter=100, mute=False):
        """
        Tecnica de Ajuste fino de hiperparametros.

        Model: Modelo a Optimizar.

        params: diccionario de parametros con el grid.

        scoring: Metricas. scoring = {'AUC': 'roc_auc', 'Accuracy': acc_scorer}
            - Anotador de metricas: acc_score = make_scorer(accuracy_score, mean_squared_error)

        refit: Metrica de importancia para optimizar el modelo'Accuracy'
        """
        acc_scorer = make_scorer(accuracy_score, mean_squared_error)
        scoring = {
            "AUC": "roc_auc",
            "Accuracy": acc_scorer
            }

        self._pre_model = model

        self.scoring = scoring

        if randomized:
            self.grid_obj = RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv, verbose=verbose,
                                                random_state=self.random_state, n_jobs=-1)
        else:
            self.grid_obj = GridSearchCV(model, params, scoring=scoring, cv=cv, refit=refit,
                                            return_train_score=True, verbose=verbose)

        self.grid_obj.fit(X, y)
        _best_Parameters = self.grid_obj.best_params_

        _results_df = pd.DataFrame(self.grid_obj.cv_results_)

        if not mute:
            print('=='*38)
            print("Mejores parametros obtenidos en el Fine-Tune:")
            print(_best_Parameters)
            print('=='*38)

        return _best_Parameters, _results_df

    def eval_train(self, model, X, y, name="Performance"):

        predictions = model.predict(X)
        errors = abs(predictions - y)
        mape = 100 * np.mean(errors / y)
        accuracy = 100 - mape
        print('=='*50)
        print('Model ' + name)
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        print('=='*50)
        print("\n")

        return accuracy

    def replace_multiclass(self, targets):

        _unic = targets.unique().tolist()
        _unic = list(np.sort(_unic))
        _remp = np.arange(0, len(_unic)).tolist()
        return targets.replace(_unic, _remp), _unic

    def eval_FineTune(self, X, y):
        """
        https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb

        """
        base_accuracy = self.eval_train(self._pre_model, X, y, name='FineTune')
        new_accuracy = self.eval_train(self.model, X, y, name='Inicial')
        print('Mejora del {:0.2f}%.'.format(100 * (new_accuracy - base_accuracy)/base_accuracy))

    def GridSearchCV_Evaluating(self, model, param, max_param, min_score=0.5):
        """
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
        """
        plt.figure(figsize=(13, 13))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

        plt.xlabel("n_estimators")
        plt.ylabel("Score")

        ax = plt.gca()
        ax.set_xlim(0, max_param)
        ax.set_ylim(min_score, 1)

        results = self.grid_obj.cv_results_

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results[param[0]].data, dtype=float)

        for scorer, color in zip(sorted(self.scoring), ['g', 'k']):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]
                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)
                ax.plot(X_axis, sample_score_mean, style, color=color,
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()

    def evaluate(self, model, X, y):
        """
        Evalucion del modelo Fine-Tune
        """
        # print(model.get_params())
        print('=='*35)
        predictions = model.predict(X)
        preds = [np.argmax(line) for line in predictions]
        report = classification_report(y, preds)
        score = accuracy_score(y_true=y, y_pred=preds)

        print(report)
        print('=='*35)
        print("{} {:0.2f}%".format("Accuracy Score evaluation : ", score*100))

    def heatmap_params(self, parameters, results_df, metric='mean_test_Accuracy'):
        """
        parametres a relacionar:
            parameters = ["n_estimators", "min_samples_split"]
        """
        x = len(self.params[parameters[0]])
        y = len(self.params[parameters[1]])

        scores = np.array(results_df[metric].values).reshape(x, y)

        sns.heatmap(scores, annot=True,
                        xticklabels=self.params[parameters[0]],
                        yticklabels=self.params[parameters[1]])

    def features_important(self, X, y, logdir="", display=True, save_image=False):
        """
        Explorar las features mas significativas
        """
        # features:
        features = X.columns.tolist()
        # Get numerical feature importances
        importances = list(self.model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

        if display or save_image:
            # Set the style
            plt.style.use('seaborn')
            # list of x locations for plotting
            x_values = list(range(len(importances)))
            # Make a bar chart
            plt.bar(x_values, importances, orientation='vertical')
            # Tick labels for x axis
            plt.xticks(x_values, features, rotation='vertical')
            # Axis labels and title
            plt.ylabel('Importancia')
            plt.xlabel('Variables')
            plt.title('Importancia de las variables')

        if display:
            plt.show()
        if save_image:
            plt.savefig(logdir)
