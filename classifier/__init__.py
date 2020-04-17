# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from .perceptron import Perceptron
from .adaline import Adaline
from .logistic_regression import LogisticRegression
from .softmax_regression import SoftmaxRegression
from .multilayerperceptron import MultiLayerPerceptron
from .ensemble_vote import EnsembleVoteClassifier
from .stacking_classification import StackingClassifier
from .stacking_cv_classification import StackingCVClassifier

__all__ = ["Perceptron", "Adaline",
           "LogisticRegression", "SoftmaxRegression",
           "MultiLayerPerceptron",
           "EnsembleVoteClassifier", "StackingClassifier",
           "StackingCVClassifier"]
