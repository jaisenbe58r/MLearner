---
title: mlearner 0.0.5dev0
subtitle: Library Documentation
author: Sebastian Raschka
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[LO,LE]{\thepage}
    - \fancyfoot[CE,CO]{}
---


![](.\sources\./img/barra.jpg)
<hr>
## Welcome to MLearner's documentation!

MLearner is a Python library of useful tools for the day-to-day data science tasks.

<hr>


![Python](.\sources\https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Build status](https://ci.appveyor.com/api/projects/status/7vx20e0h5dxcyla2/branch/master?svg=true)](https://ci.appveyor.com/project/jaisenbe58r/MLearner/branch/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/68209df46c8240b887db5ae5fa3cb410)](https://www.codacy.com/manual/jaisenbe58r/MLearner?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jaisenbe58r/MLearner&amp;utm_campaign=Badge_Grade)
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



# `utils.Counter`
## hola

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
pip install git+git://github.com/jaisenbe58r/mlearner.git
```

Or, you can fork the GitHub repository from https://github.com/jaisenbe58r/MLearner and install mlearner from your local drive via

```bash
python setup.py install
```


# Release Notes

---

The CHANGELOG for the current development version is available at
[https://github.com/jaisenbe58r/MLearner/blob/master/docs/sources/CHANGELOG.md](https://github.com/jaisenbe58r/MLearner/blob/master/docs/sources/CHANGELOG.md).

---



### Version 0.1.0 (2020-04-17)

- First Version


# How to Contribute

---

I would be very happy about any kind of contributions that help to improve and extend the functionality of mlearner.


## Quick Contributor Checklist

This is a quick checklist about the different steps of a typical contribution to mlearner (and
other open source projects). Consider copying this list to a local text file (or the issue tracker)
and checking off items as you go.

1. [ ]  Open a new "issue" on GitHub to discuss the new feature / bug fix  
2. [ ]  Fork the mlearner repository from GitHub (if not already done earlier)
3. [ ]  Create and check out a new topic branch (please don't make modifications in the master branch)
4. [ ]  Implement the new feature or apply the bug-fix  
5. [ ]  Add appropriate unit test functions in `mlearner/*/tests`
6. [ ]  Run `PYTHONPATH='.' pytest ./mlearner -sv` and make sure that all unit tests pass  
7. [ ]  Check for style issues by running `flake8 ./mlearner` (you may want to run `pytest` again after you made modifications to the code)
8. [ ]  Add a note about the modification/contribution to the `./docs/sources/changelog.md` file  
9. [ ]  Modify documentation in the appropriate location under `mlearner/docs/sources/`  
10. [ ]  Push the topic branch to the server and create a pull request
11. [ ]  Check the Travis-CI build passed at [https://travis-ci.org/rasbt/mlearner](https://travis-ci.org/rasbt/mlearner)
12. [ ]  Check/improve the unit test coverage at [https://coveralls.io/github/rasbt/mlearner](https://coveralls.io/github/rasbt/mlearner)
13. [ ]  Check/improve the code health at [https://landscape.io/github/rasbt/mlearner](https://landscape.io/github/rasbt/mlearner)

<hr>

# Tips for Contributors


## Getting Started - Creating a New Issue and Forking the Repository

- If you don't have a [GitHub](https://github.com) account, yet, please create one to contribute to this project.
- Please submit a ticket for your issue to discuss the fix or new feature before too much time and effort is spent for the implementation.

![](.\sources\./img/contributing/new_issue.png)

- Fork the `mlearner` repository from the GitHub web interface.

![](.\sources\./img/contributing/fork.png)

- Clone the `mlearner` repository to your local machine by executing
 ```git clone https://github.com/<your_username>/mlearner.git```

## Syncing an Existing Fork

If you already forked mlearner earlier, you can bring you "Fork" up to date
with the master branch as follows:

#### 1. Configuring a remote that points to the upstream repository on GitHub

List the current configured remote repository of your fork by executing

```bash
$ git remote -v
```

If you see something like

```bash
origin	https://github.com/<your username>/mlearner.git (fetch)
origin	https://github.com/<your username>/mlearner.git (push)
```
you need to specify a new remote *upstream* repository via

```bash
$ git remote add upstream https://github.com/jaisenbe58r/MLearner.git
```

Now, verify the new upstream repository you've specified for your fork by executing

```bash
$ git remote -v
```

You should see following output if everything is configured correctly:

```bash
origin	https://github.com/<your username>/mlearner.git (fetch)
origin	https://github.com/<your username>/mlearner.git (push)
upstream	https://github.com/jaisenbe58r/MLearner.git (fetch)
upstream	https://github.com/jaisenbe58r/MLearner.git (push)
```

#### 2. Syncing your Fork

First, fetch the updates of the original project's master branch by executing:

```bash
$ git fetch upstream
```

You should see the following output

```bash
remote: Counting objects: xx, done.
remote: Compressing objects: 100% (xx/xx), done.
remote: Total xx (delta xx), reused xx (delta x)
Unpacking objects: 100% (xx/xx), done.
From https://github.com/jaisenbe58r/MLearner
 * [new branch]      master     -> upstream/master
```

This means that the commits to the `rasbt/mlearner` master branch are now
stored in the local branch `upstream/master`.

If you are not already on your local project's master branch, execute

```bash
$ git checkout master
```

Finally, merge the changes in upstream/master to your local master branch by
executing

```bash
$ git merge upstream/master
```

which will give you an output that looks similar to

```bash
Updating xxx...xxx
Fast-forward
SOME FILE1                    |    12 +++++++
SOME FILE2                    |    10 +++++++
2 files changed, 22 insertions(+),
```


## *The Main Workflow - Making Changes in a New Topic Branch

Listed below are the 9 typical steps of a contribution.

#### 1. Discussing the Feature or Modification

Before you start coding, please discuss the new feature, bugfix, or other modification to the project
on the project's [issue tracker](https://github.com/jaisenbe58r/MLearner/issues). Before you open a "new issue," please
do a quick search to see if a similar issue has been submitted already.

#### 2. Creating a new feature branch

Please avoid working directly on the master branch but create a new feature branch:

```bash
$ git branch <new_feature>
```

Switch to the new feature branch by executing

```bash
$ git checkout <new_feature>
```

#### 3. Developing the new feature / bug fix

Now it's time to modify existing code or to contribute new code to the project.

#### 4. Testing your code

Add the respective unit tests and check if they pass:

```bash
$ PYTHONPATH='.' pytest ./mlearner ---with-coverage
```


#### 5. Documenting changes

Please add an entry to the `mlearner/docs/sources/changelog.md` file.
If it is a new feature, it would also be nice if you could update the documentation in appropriate location in `mlearner/sources`.


#### 6. Committing changes

When you are ready to commit the changes, please provide a meaningful `commit` message:

```bash
$ git add <modifies_files> # or `git add .`
$ git commit -m '<meaningful commit message>'
```

#### 7. Optional: squashing commits

If you made multiple smaller commits, it would be nice if you could group them into a larger, summarizing commit. First, list your recent commit via

**Note**  
**Due to the improved GitHub UI, this is no longer necessary/encouraged.**


```bash
$ git log
```

which will list the commits from newest to oldest in the following format by default:


```bash
commit 046e3af8a9127df8eac879454f029937c8a31c41
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 03:46:37 2015 -0500

    fixed setup.py

commit c3c00f6ba0e8f48bbe1c9081b8ae3817e57ecc5c
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 03:04:39 2015 -0500

        documented feature x

commit d87934fe8726c46f0b166d6290a3bf38915d6e75
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 02:44:45 2015 -0500

        added support for feature x
```

Assuming that it would make sense to group these 3 commits into one, we can execute

```bash
$ git rebase -i HEAD~3
```

which will bring our default git editor with the following contents:

```bash
pick d87934f added support for feature x
pick c3c00f6 documented feature x
pick 046e3af fixed setup.py
```

Since `c3c00f6` and `046e3af` are related to the original commit of `feature x`, let's keep the `d87934f` and squash the 2 following commits into this initial one by changes the lines to


```
pick d87934f added support for feature x
squash c3c00f6 documented feature x
squash 046e3af fixed setup.py
```

Now, save the changes in your editor. Now, quitting the editor will apply the `rebase` changes, and the editor will open a second time, prompting you to enter a new commit message. In this case, we could enter `support for feature x` to summarize the contributions.


#### 8. Uploading changes

Push your changes to a topic branch to the git server by executing:

```bash
$ git push origin <feature_branch>
```

#### 9. Submitting a `pull request`

Go to your GitHub repository online, select the new feature branch, and submit a new pull request:


![](.\sources\./img/contributing/pull_request.png)


<hr>

# Notes for Developers



## Building the documentation

The documentation is built via [MkDocs](http://www.mkdocs.org); to ensure that the documentation is rendered correctly, you can view the documentation locally by executing `mkdocs serve` from the `mlearner/docs` directory.

For example,

```bash
~/github/mlearner/docs$ mkdocs serve
```

### 1. Building the API documentation

To build the API documentation, navigate to `mlearner/docs` and execute the `make_api.py` file from this directory via

```python
~/github/mlearner/docs$ python make_api.py
```

This should place the API documentation into the correct directories into the two directories:

- `mlearner/docs/sources/api_modules`
- `mlearner/docs/sources/api_subpackes`

### 2. Editing the User Guide

The documents containing code examples for the "User Guide" are generated from IPython Notebook files. In order to convert a IPython notebook file to markdown after editing, please follow the following steps:

1. Modify or edit the existing notebook.
2. Execute all cells in the current notebook and make sure that no errors occur.
3. Convert the notebook to markdown using the `ipynb2markdown.py` converter

```python
~/github/mlearner/docs$ python ipynb2markdown.py --ipynb_path ./sources/user_guide/subpackage/notebookname.ipynb
```

**Note**  

If you are adding a new document, please also include it in the pages section in the `mlearner/docs/mkdocs.yml` file.



### 3. Building static HTML files of the documentation

First, please check the documenation via localhost (http://127.0.0.1:8000/):

```bash
~/github/mlearner/docs$ mkdocs serve
```

Next, build the static HTML files of the mlearner documentation via

```bash
~/github/mlearner/docs$ mkdocs build --clean
```

To deploy the documentation, execute

```bash
~/github/mlearner/docs$ mkdocs gh-deploy --clean
```

### 4. Generate a PDF of the documentation

To generate a PDF version of the documentation, simply `cd` into the `mlearner/docs` directory and execute:

```bash
python md2pdf.py
```

## Uploading a new version to PyPI

### 1. Creating a new testing environment

Assuming we are using `conda`, create a new python environment via

```bash
$ conda create -n 'mlearner-testing' python=3 numpy scipy pandas
```

Next, activate the environment by executing

```bash
$ source activate mlearner-testing
```

### 2. Installing the package from local files

Test the installation by executing

```bash
$ python setup.py install --record files.txt
```

the `--record files.txt` flag will create a `files.txt` file listing the locations where these files will be installed.


Try to import the package to see if it works, for example, by executing

```bash
$ python -c 'import mlearner; print(mlearner.__file__)'
```

If everything seems to be fine, remove the installation via

```bash
$ cat files.txt | xargs rm -rf ; rm files.txt
```

Next, test if `pip` is able to install the packages. First, navigate to a different directory, and from there, install the package:

```bash
$ pip install mlearner
```

and uninstall it again

```bash
$ pip uninstall mlearner
```

### 3. Deploying the package

Consider deploying the package to the PyPI test server first. The setup instructions can be found [here](https://wiki.python.org/moin/TestPyPI).

```bash
$ python setup.py sdist bdist_wheel upload -r https://testpypi.python.org/pypi
```

Test if it can be installed from there by executing

```bash
$ pip install -i https://testpypi.python.org/pypi mlearner
```

and uninstall it

```bash
$ pip uninstall mlearner
```

After this dry-run succeeded, repeat this process using the "real" PyPI:

```bash
$ python setup.py sdist bdist_wheel upload
```

### 4. Removing the virtual environment

Finally, to cleanup our local drive, remove the virtual testing environment via

```bash
$ conda remove --name 'mlearner-testing' --all
```

### 5. Updating the conda-forge recipe

Once a new version of mlearner has been uploaded to PyPI, update the conda-forge build recipe at https://github.com/conda-forge/mlearner-feedstock by changing the version number in the `recipe/meta.yaml` file appropriately.


# Contributors

For the current list of contributors to mlearner, please see the GitHub contributor page at [https://github.com/jaisenbe58r/MLearner/graphs/contributors].


- This project is released under a permissive new BSD open source license and commercially usable. There is no warranty; not even for merchantability or fitness for a particular purpose.

- In addition, you may use, copy, modify, and redistribute all artistic creative works (figures and images) included in this distribution under the directory 
according to the terms and conditions of the Creative Commons Attribution 4.0 International License. (Computer-generated graphics such as the plots produced by matplotlib fall under the BSD license mentioned above).


# new BSD License

---

New BSD License

Copyright (c) 2014-2020, Sebastian Raschka. All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of mlearner nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Creative Commons Attribution 4.0 International License


mlearner documentation figures are licensed under a
Creative Commons Attribution 4.0 International License.

<http://creativecommons.org/licenses/by-sa/4.0/>.

#### You are free to:


Share — copy and redistribute the material in any medium or format
Adapt — remix, transform, and build upon the material
for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.

#### Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.


# Citing mlearner



# Discuss


Any questions or comments about mlearner? Join the mlearner mailing list on Google Groups!



