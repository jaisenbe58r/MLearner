
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


class EvaluationModels(object):
    def __init__(self, model, random_state=99):
        """
        Inicializacion e la clase de Preprocesado de un dataframe
        """
        self.model = model
        self.random_state = random_state

    def add_model(self, filename):
        """
        Load the model from disk
        """
        self.model = pickle.load(open(filename, 'rb'))

    def restore_model(self, filename):
        """
        Load the model from disk
        """
        self.model = pickle.load(open(filename, 'rb'))

    def save_model(self, filename):
        """
        Save the model to disk
        """
        pickle.dump(self.model, open(filename, 'wb'))

    def confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def validacion_cruzada(self, X, Y, n_splits, shuffle=True, scoring="accuracy"):
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
        scores = cross_val_score(self.model, X, Y, scoring=scoring, cv=cv)

        return scores.mean()

    def evaluacion_rf_2features(self, clf, data_eval, data_eval_target, targets=[0, 1], save=False, logdir_report="", display=True):
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

        probs = clf.predict_proba(data_eval)
        prob = probs[:, 1]
        df = pd.DataFrame(prob)
        df = df.set_index(np.array(list(data_eval.index)))

        data_remp = data_eval_target.copy()
        data_remp = data_remp.replace({targets[1]: 1, targets[0]: 0})

        _, _, thresholders = roc_curve(data_remp, prob)

        for i in range(len(thresholders)):

            df["prediction"] = np.where(df[0] >= thresholders[i], targets[1], targets[0])
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

        df["prediction"] = np.where(df[0] >= thresholders[index], targets[1], targets[0])
        df["actual"] = list(data_eval_target)
        df["result"] = np.where(df["prediction"] == df["actual"], 1, 0)

        return th, res, df

    def create_ROC(self, lm, X_test, Y_test, targets=[0, 1], logdir_report="", display=True, save_image=True):
        """
        Se crea la curva ROC de las predicciones de conjunto de test.

        Outputs:
            - df: Dataframe de los datos de metricas ROC.
            - auc: Area por debajo de la curfa ROC (efectividad de las predicciones).
        """
        probs = lm.predict_proba(X_test)
        prob = probs[:, 1]

        data_remp = Y_test.copy()
        data_remp = data_remp.replace({targets[1]: 1, targets[0]: 0})

        espc_1, sensit, thresholds = roc_curve(data_remp, prob)

        df = pd.DataFrame({
            "esp": espc_1,
            "sens": sensit,
            "TH": thresholds
        })

        _auc = auc(espc_1, sensit)

        if display:

            plt.plot(df.esp, df.sens, marker="o", linestyle="--", color="r")
            x = [i*0.01 for i in range(100)]
            y = [i*0.01 for i in range(100)]
            plt.plot(x, y)
            plt.xlabel("1-Especifidad")
            plt.ylabel("Sensibilidad")
            plt.title("Curva ROC")

            if save_image:
                plt.savefig(logdir_report + "_ROC.png")

            # plt.close()

        return df, _auc

    def create_ROC_pro(self, fprs, tprs, X, y, targets=[0, 1], logdir_report="", display=True, save_image=True):
        """
        Plot the Receiver Operating Characteristic from a list
        of true positive rates and false positive rates.
        """
        # Initialize useful lists + the plot axes.
        tprs_interp = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        f, ax = plt.subplots(figsize=(14, 10))

        # remplazo de datos a [0, 1]
        data_remp = y.copy()
        y = data_remp.replace({targets[1]: 1, targets[0]: 0})

        # Plot ROC for each K-Fold + compute AUC scores.
        for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
            tprs_interp[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

        # Plot the luck line.
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

        # Plot the mean ROC.
        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        # Plot the standard deviation around the mean ROC.
        std_tpr = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        # Fine tune and show the plot.
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")

        if display:
            plt.show()
        if save_image:
            plt.savefig(logdir_report + "_ROC_pro.png")

            plt.close()

        return (f, ax)

    def compute_roc_auc(self, model, index, X, y):
        """
        Computo para todas las pruebas de KFold
        """
        y_predict = model.predict_proba(X.iloc[index])[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[index], y_predict)
        auc_score = auc(fpr, tpr)

        return fpr, tpr, auc_score

    def class_report(self, y_true, predictions, clases, save=False, logdir_report=""):
        """
        Un informe de clasificacion se utiliza para medir la calidad de las predicciones de un
        algoritmo de clasificacion. Cuántas predicciones son verdaderas y cuántas son falsas.
        Más especificamente, los Positivos verdaderos, los Positivos falsos, los negativos
        verdaderos y los negativos falsos se utilizan para predecir las metricas de un informe
        de clasificacion.

        El informe muestra las principales metricas de clasificacion de precision, recuperacion
        y puntaje f1 por clase. Las metricas se calculan utilizando verdaderos y falsos positivos,
        verdaderos y falsos negativos. Positivo y negativo en este caso son nombres genericos
        para las clases predichas. Hay cuatro formas de verificar si las predicciones son
        correctas o incorrectas:

            TN / Verdadero negativo: cuando un caso fue negativo y se pronostico negativo
            TP / Verdadero Positivo: cuando un caso fue positivo y predicho positivo
            FN / Falso negativo: cuando un caso fue positivo pero predicho negativo
            FP / Falso Positivo: cuando un caso fue negativo pero predicho positivo

        La precision es la capacidad de un clasificador de no etiquetar una instancia positiva
        que en realidad es negativa. Para cada clase se define como la relacion de positivos
        verdaderos a la suma de positivos verdaderos y falsos.

            TP - Positivos verdaderos
            FP - Positivos falsos

            Precision: precision de las predicciones positivas.
            Precision = TP / (TP + FP)

        Recordar es la capacidad de un clasificador para encontrar todas las instancias positivas.
        Para cada clase se define como la relacion entre los verdaderos positivos y la suma de
        los verdaderos positivos y los falsos negativos.

            FN - Falsos negativos

            Recordar: fraccion de positivos identificados correctamente.
            Recuperacion = TP / (TP + FN)

        El  puntaje F 1 es una media armonica ponderada de precision y recuperacion de modo que
        el mejor puntaje es 1.0 y el peor es 0.0. En terminos generales, los  puntajes de F 1 son
        más bajos que las medidas de precision, ya que incorporan precision y recuerdo en su
        cálculo. Como regla general, el promedio ponderado de F 1  debe usarse para comparar
        modelos clasificadores, no la precision global.

            Puntuacion F1 = 2 * (recuperacion * precision) / (recuperacion + precision)
        """
        target_names = ["{}".format(clases[i]) for i in range(len(clases))]
        cr = classification_report(y_true, predictions, target_names=target_names)
        print("\nInforme de clasificacion:\n")
        print(cr)
        if save:
            print(cr, file=open(logdir_report + "_Informe_Clasificacion" + ".txt", 'w'))

        return cr

    def plot_Histograma(self, predict, correct, incorrect, logdir_report, categorias=[0, 1],
                        display=True, save_image=True):
        import tensorflow as tf
        i_pred_correct = tf.argmax(predict[correct], axis=1, output_type=tf.int32)
        i_pred_incorrect = tf.argmax(predict[incorrect], axis=1, output_type=tf.int32)

        plt.figure()
        _, ax = plt.subplots(2, 1)

        ax[0].set_ylabel('frequencia')
        ax[0].set_xlabel('valores')
        ax[0].legend(['Seeds', 'Craters'])
        ax[0].set_title('Histograma - Predicciones correctas')

        ax[1].set_ylabel('frequencia')
        ax[1].set_xlabel('valores')
        ax[1].legend(['Seeds', 'Craters'])
        ax[1].set_title('Histograma - Predicciones incorrectas')

        for i in categorias:
            i_pred_correct = np.nonzero(i_pred_correct == i)[0]
            ax[0].hist(predict[i_pred_correct][i_pred_correct, 0], alpha=0.8)
            i_pred_incorrect = np.nonzero(i_pred_incorrect == i)[0]
            ax[1].hist(predict[incorrect][i_pred_incorrect, 0], alpha=0.8)

        plt.tight_layout()

        if display:
            plt.show()

        if save_image:
            plt.savefig(logdir_report + "_histogram.png")

    def plot_confusion_matrix(self, y_true, y_pred, classes, num_clases, logdir_report,
                                normalize=False, title=None, cmap=cm.Blues, name="cm_normalizada"):
        """
        Una matriz de confusion es un resumen de los resultados de prediccion sobre un problema
        de clasificacion.

        El numero de predicciones correctas e incorrectas se resume con valores de conteo y se
        desglosa por clase. Esta es la clave de la matriz de confusion.

        La matriz de confusion muestra las formas en que su modelo de clasificacion
        se confunde cuando hace predicciones.

        Le da una idea no solo de los errores que está cometiendo su clasificador, sino más
        importante aun, de los tipos de errores que se están cometiendo.

        Es este desglose el que supera la limitacion del uso de la precision de clasificacion solo.
        """

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = "Matriz de confusion normalizada"
            else:
                title = "Matriz de confusion sin normalizar"
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[0:num_clases]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("\nMatriz de confusion normalizada:\n")
        else:
            print("\nMatriz de confusion sin normalizar:\n")

        print(cm)

        fig, ax = plt.subplots()

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="Clase real",
            xlabel="Clase predecida",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        plt.savefig(logdir_report + "_" + name + "_cm.png")
        # plt.show()
        plt.close()
        return ax
