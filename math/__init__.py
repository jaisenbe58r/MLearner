# Jaime Sendra Berenguer-2020
# MLearner Machine Learning Library Extensions
# Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
#
# License: MIT

from .counting import num_combinations
from .counting import num_permutations
from .counting import factorial
from .linalg import vectorspace_orthonormalization
from .linalg import vectorspace_dimensionality

__all__ = ["num_combinations", "num_permutations",
           "factorial", "vectorspace_orthonormalization",
           "vectorspace_dimensionality"]
