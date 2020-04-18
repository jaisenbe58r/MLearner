# from distutils.core import setup
from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import mlearner

VERSION = mlearner.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')


def readme():
    with open("README.md", "r") as fh:
        return fh.read()


setup(
  name='mlearner',         # How you named your package folder (MyLib)
  packages=find_packages(),   # Chose the same as "name"
  version=VERSION,      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='Machine Learning Library Extensions',   # Give a short description about your library
  author='Jaime Sendra',                   # Type in your name
  author_email='jaisenberafel@gmail.com',      # Type in your E-Mail
  url='https://github.com/jaisenbe58r/MLearner',   # Provide either the link to your github or to your website
  download_url='https://github.com/jaisenbe58r/MLearner/archive/v0.0.2.tar.gz',    # I explain this later on
  keywords=['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
  package_data={'': [
                    'README.md',
                    'requirements.txt']
                },
  include_package_data=True,
  install_requires=install_reqs,
  platforms='any',

  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      # Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Image Recognition',
  ],
    python_requires='>=3.6',
    long_description=readme(),
    long_description_content_type="text/markdown",
)

# setup(
#   name = 'mlearner',         # How you named your package folder (MyLib)
#   packages = ['mlearner'],   # Chose the same as "name"
#   version = VERSION,      # Start with a small number and increase it with every change you make
#   license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
#   description = 'Machine Learning Library Extensions',   # Give a short description about your library
#   author = 'Jaime Sendra Berenguer',                   # Type in your name
#   author_email = 'jaisenberafel@gmail.com',      # Type in your E-Mail
#   url = 'https://github.com/jaisenbe58r/MLearner',   # Provide either the link to your github or to your website
#   download_url = 'https://github.com/jaisenbe58r/MLearner/archive/v0.0.2.tar.gz',    # I explain this later on
#   keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
#   install_requires=[            # I get to this in a second
#           'validators',
#           'beautifulsoup4',
#       ],
#   classifiers=[
#     'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
#     'Intended Audience :: Developers',      # Define that your audience are developers
#     'Topic :: Software Development :: Build Tools',
#     'License :: OSI Approved :: MIT License',   # Again, pick a license
#     'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
#     'Programming Language :: Python :: 3.4',
#     'Programming Language :: Python :: 3.5',
#     'Programming Language :: Python :: 3.6',
#     ],
#     python_requires='>=3.6',
#     ong_description=long_description,
#     long_description_content_type="text/markdown",
# )