# Installing mlearn

---

### PyPI

To install mlearn, just execute  

```bash
pip install mlearn  
```

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlearn](https://pypi.python.org/pypi/mlearn), unzip it, navigate into the package, and use the command:

```bash
python setup.py install
```

##### Upgrading via `pip`

To upgrade an existing version of mlearn from PyPI, execute

```bash
pip install mlearn --upgrade --no-deps
```

Please note that the dependencies (NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

##### Installing mlearn from the source distribution

In rare cases, users reported problems on certain systems with the default `pip` installation command, which installs mlearn from the binary distribution ("wheels") on PyPI. If you should encounter similar problems, you could try to install mlearn from the source distribution instead via

```bash
pip install --no-binary :all: mlearn
```

Also, I would appreciate it if you could report any issues that occur when using `pip install mlearn` in hope that we can fix these in future releases.

### Conda

The mlearn package is also [available through conda forge](https://github.com/conda-forge/mlearn-feedstock). 

To install mlearn using conda, use the following command:

    conda install mlearn --channel conda-forge

or simply 

    conda install mlearn

if you added conda-forge to your channels (`conda config --add channels conda-forge`).

### Dev Version

The mlearn version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/jaisenbe58r/mlearn.git
```

Or, you can fork the GitHub repository from https://github.com/jaisenbe58r/MLearn and install mlearn from your local drive via

```bash
python setup.py install
```
