---
title: Mlxtend 0.0.5dev0
subtitle: Library Documentation
author: Sebastian Raschka
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[LO,LE]{\thepage}
    - \fancyfoot[CE,CO]{}
---


![](.\sources\./img/Carpeta.PNG)
## Welcome to MLearner's documentation!

MLearner is a Python library of useful tools for the day-to-day data science tasks.

<hr>


![Python](.\sources\https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Build status](https://ci.appveyor.com/api/projects/status/7vx20e0h5dxcyla2/branch/master?svg=true)](https://ci.appveyor.com/project/jaisenbe58r/MLearner/branch/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2f5b0302acc04a3dac74d6815fdf66e5)](https://www.codacy.com/manual/jaisenbe58r/MLearner?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jaisenbe58r/MLearner&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

<hr>

## Links

- **Documentation:** [https://jaisenbe58r.github.io/MLearner/](https://jaisenbe58r.github.io/MLearner/)
- **Source code repository:** [https://github.com/jaisenbe58r/MLearner](https://github.com/jaisenbe58r/MLearner)
- **PyPI:** [https://pypi.python.org/pypi/mlearner](https://pypi.python.org/pypi/mlearner)
- **Questions?** Check out the [Discord group MLearner](https://discordapp.com/invite/HUxahg)

<hr>

## Examples

```python

```


## License

MIT License

Copyright (c) 2020 Jaime Sendra Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Contact

I received a lot of feedback and questions about mlearner recently, and I thought that it would be worthwhile to set up a public communication channel. Before you write an email with a question about mlearner, please consider posting it here since it can also be useful to others! Please join the [Discord group MLearner](https://discordapp.com/invite/HUxahg)

If Google Groups is not for you, please feel free to write me an [email](mailto:jaisenberafel@gmail.com) or consider filing an issue on [GitHub's issue tracker](https://github.com/jaisenbe58r/MLearner/issues) for new feature requests or bug reports. In addition, I setup a [Gitter channel](https://gitter.im/coMLearner/community#) for live discussions.



# `data.wine_data`
## wine_data

*wine_data()*

Wine dataset.

Source: https://archive.ics.uci.edu/ml/datasets/Wine
Number of samples: 178
Class labels: {0, 1, 2}, distribution: [59, 71, 48]

Data Set Information:

These data are the results of a chemical analysis of wines grown
in the same region in Italy but derived from three different cultivars.
The analysis determined the quantities of 13 constituents found in each
of the three types of wines.

The attributes are (dontated by Riccardo Leardi, riclea@anchem.unige.it)

- 1) Alcohol
- 2) Malic acid
- 3) Ash
- 4) Alcalinity of ash
- 5) Magnesium
- 6) Total phenols
- 7) Flavanoids
- 8) Nonflavanoid phenols
- 9) Proanthocyanins
- 10) Color intensity
- 11) Hue
- 12) OD280/OD315 of diluted wines
- 13) Proline

In a classification context, this is a well posed problem with "well behaved"
class structures. A good data set for first testing of a new classifier,
but not very challenging.

**Returns**

- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 178 wine samples as rows
    and 13 feature columns.
    y is a 1-dimensional array of the 3 class labels 0, 1, 2

**Examples**

For usage examples, please see
    https://jaisenbe58r.github.io/MLearner/user_guide/data/wine_data



# `preprocessing.MeanCenterer`

A transformer object that performs column-based mean centering on a NumPy array.

> from mlearner.preprocessing import MeanCenterer

## Example 1 - Centering a NumPy Array

Use the `fit` method to fit the column means of a dataset (e.g., the training dataset) to a new `MeanCenterer` object. Then, call the `transform` method on the same dataset to center it at the sample mean.


```python
import numpy as np
from mlearner.preprocessing import MeanCenterer
X_train = np.array(
                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
mc = MeanCenterer().fit(X_train)
mc.transform(X_train)
```




    array([[-3., -3., -3.],
           [ 0.,  0.,  0.],
           [ 3.,  3.,  3.]])



## API


*MeanCenterer()*

Column centering of pandas Dataframeself.

**Attributes**

col_means:  numpy.ndarray [n_columns] or pandas [n_columns]
mean values for centering after fitting the MeanCenterer object.

**Examples**

For usage examples, please see
https://jaisenbe58r.github.io/MLearner/user_guide/data/MeanCenterer/

### Methods

<hr>

*fit(X, y=None)*

Gets the column means for mean centering.

**Parameters**

- `X` : {Dataframe}, shape = [n_samples, n_features]

    Dataframe, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

self

<hr>

*fit_transform(X, y=None, **fit_params)*

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

**Parameters**

- `X` : numpy array of shape [n_samples, n_features]

    Training set.


- `y` : numpy array of shape [n_samples]

    Target values.

**Returns**

- `X_new` : numpy array of shape [n_samples, n_features_new]

    Transformed array.

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X)*

Centers a pandas.

**Parameters**

- `X` : {Dataframe}, shape = [n_samples, n_features]

    Dataframe of samples, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `X_transform` : {DAtaframe}, shape = [n_samples, n_features]

    A copy of the input Dataframe with the columns centered.



ython


