# Installing mlearner

---

### PyPI

To install mlearner, just execute  

```bash
pip install mlearner  
```

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlearner](https://pypi.python.org/pypi/mlearner), unzip it, navigate into the package, and use the command:

```bash
python setup.py install
```

##### Upgrading via `pip`

To upgrade an existing version of mlearner from PyPI, execute

```bash
pip install mlearner --upgrade --no-deps
```

Please note that the dependencies (NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

##### Installing mlearner from the source distribution

In rare cases, users reported problems on certain systems with the default `pip` installation command, which installs mlearner from the binary distribution ("wheels") on PyPI. If you should encounter similar problems, you could try to install mlearner from the source distribution instead via

```bash
pip install --no-binary :all: mlearner
```

Also, I would appreciate it if you could report any issues that occur when using `pip install mlearner` in hope that we can fix these in future releases.

### Conda

The mlearner package is also [available through conda forge](https://github.com/conda-forge/mlearner-feedstock). 

To install mlearner using conda, use the following command:

    conda install mlearner --channel conda-forge

or simply 

    conda install mlearner

if you added conda-forge to your channels (`conda config --add channels conda-forge`).

### Dev Version

The mlearner version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/rasbt/mlearner.git
```

Or, you can fork the GitHub repository from https://github.com/jaisenbe58r/MLearner and install mlearner from your local drive via

```bash
python setup.py install
```
