
![](./img/barra.jpg)

### Welcome to mlearner's documentation!

**mlearner (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.**



[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)](https://code.visualstudio.com/)
![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![GitHub release](https://img.shields.io/github/release/jaisenbe58r/mlearner.svg)](https://GitHub.com/jaisenbe58r/mlearner/releases/)
[![GitHub commits](https://img.shields.io/github/commits-since/jaisenbe58r/mlearner/v0.0.5.svg)](https://GitHub.com/jaisenbe58r/mlearner/commit/)
[![PyPI Latest Release](https://badge.fury.io/py/matplotlib.svg)](https://pypi.org/project/mlearner/)
[![Build status](https://ci.appveyor.com/api/projects/status/7vx20e0h5dxcyla2/branch/master?svg=true)](https://ci.appveyor.com/project/jaisenbe58r/mlearner/branch/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/68209df46c8240b887db5ae5fa3cb410)](https://www.codacy.com/manual/jaisenbe58r/MLearner?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jaisenbe58r/MLearner&amp;utm_campaign=Badge_Grade)
[![Build Status](https://travis-ci.org/jaisenbe58r/mlearner.svg?branch=master)](https://travis-ci.org/jaisenbe58r/mlearner)
[![codecov](https://codecov.io/gh/jaisenbe58r/MLearner/branch/master/graph/badge.svg)](https://codecov.io/gh/jaisenbe58r/MLearner)
[![Coverage Status](https://coveralls.io/repos/github/jaisenbe58r/MLearner/badge.svg?branch=master)](https://coveralls.io/github/jaisenbe58r/MLearner?branch=master)
[![License](https://img.shields.io/badge/license-MIT-ORANGE.svg)](https://github.com/jaisenbe58r/mlearner/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/256283484.svg)](https://zenodo.org/badge/latestdoi/256283484)
[![GitHub contributors](https://img.shields.io/github/contributors/jaisenbe58r/mlearner.svg)](https://GitHub.com/jaisenbe58r/mlearner/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/jaisenbe58r/mlearner.svg)](https://GitHub.com/jaisenbe58r/mlearner/issues/)
[![Discuss](https://img.shields.io/badge/discuss-DISCORD-PURPLE.svg)](https://discord.gg/HUxahg)
[![GitHubIssues](https://img.shields.io/badge/issue_tracking-github-violet.svg)](https://github.com/jaisenbe58r/MLearner/issues)
[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://saythanks.io/to/kennethreitz)



<hr>

## Links

- **Documentation:** [http://jaisenbe58r.github.io/mlearner](http://jaisenbe58r.github.io/mlearner)
- Source code repository: [https://github.com/rasbt/mlearner](https://github.com/rasbt/mlearner)
- PyPI: [https://pypi.python.org/pypi/mlearner](https://pypi.python.org/pypi/mlearner)
- Questions? Check out the [Google Groups mailing list](https://groups.google.com/forum/#!forum/mlearner)

<hr>


## Examples

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlearner.classifier import EnsembleVoteClassifier
from mlearner.data import iris_data
from mlearner.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                              weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
X = X[:,[0, 2]]

# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         labels,
                         itertools.product([0, 1],
                         repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y,
                                clf=clf, legend=2)
    plt.title(lab)

plt.show()
```

---

![](./img/ensemble_decision_regions_2d.png)

If you use mlearner as part of your workflow in a scientific publication, please consider citing the mlearner repository with the following DOI:

[![DOI](http://joss.theoj.org/papers/10.21105/joss.00638/status.svg)](https://doi.org/10.21105/joss.00638)

```
@article{raschkas_2018_mlxtend,
  author       = {Sebastian Raschka},
  title        = {mlearner: Providing machine learning and data science 
                  utilities and extensions to Python’s  
                  scientific computing stack},
  journal      = {The Journal of Open Source Software},
  volume       = {3},
  number       = {24},
  month        = apr,
  year         = 2018,
  publisher    = {The Open Journal},
  doi          = {10.21105/joss.00638},
  url          = {http://joss.theoj.org/papers/10.21105/joss.00638}
}
```


## License

- This project is released under a permissive new BSD open source license ([LICENSE-BSD3.txt](https://github.com/rasbt/mlearner/blob/master/LICENSE-BSD3.txt)) and commercially usable. There is no warranty; not even for merchantability or fitness for a particular purpose.
- In addition, you may use, copy, modify and redistribute all artistic creative works (figures and images) included in this distribution under the directory
according to the terms and conditions of the Creative Commons Attribution 4.0 International License.  See the file [LICENSE-CC-BY.txt](https://github.com/rasbt/mlearner/blob/master/LICENSE-CC-BY.txt) for details. (Computer-generated graphics such as the plots produced by matplotlib fall under the BSD license mentioned above).

## Contact

I received a lot of feedback and questions about mlearner recently, and I thought that it would be worthwhile to set up a public communication channel. Before you write an email with a question about mlearner, please consider posting it here since it can also be useful to others! Please join the [Google Groups Mailing List](https://groups.google.com/forum/#!forum/mlearner)!

If Google Groups is not for you, please feel free to write me an [email](mailto:mail@sebastianraschka.com) or consider filing an issue on [GitHub's issue tracker](https://github.com/rasbt/mlearner/issues) for new feature requests or bug reports. In addition, I setup a [Gitter channel](https://gitter.im/rasbt/mlearner?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) for live discussions.
