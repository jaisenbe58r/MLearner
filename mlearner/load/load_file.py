
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author: Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd


class DataLoad(object):

    """Loading a dataset from a file or dataframe.

    Parameters
    ----------
    data: pandas [n_columns, n_features].
        pandas Dataframe.

    Attributes
    ----------
    n: lenght of dataset.
    start: start iterator.
    end: end iterator.
    num: current iterator.

    Examples
    --------
    For usage examples, please see:
    https://jaisenbe58r.github.io/MLearner/user_guide/load/DataLoad/


    """
    def __init__(self, data):
        self.data = data
        self.n = len(self.data)
        self.start = 0
        self.end = self.n
        self.num = 0

    @classmethod
    def load_data(cls, filename, name="dataset", sep=';', decimal=",", **params):
        """Loading a dataset from a csv file.

        Parameters
        ----------
        filename: `str, path object or file-like object`
            Any valid string path is acceptable. The string could be a URL.
            Valid URL schemes include http, ftp, s3, and file. For file URLs,
            a host is expected. A local file could be:
            `file://localhost/path/to/table.csv`.
            If you want to pass in a path object, pandas accepts any os.PathLike.
            By file-like object, we refer to objects with a read() method,
            such as a file handler (e.g. via builtin open function) or StringIO.

        seps: `str`
            Delimiter to use. If sep is None, the C engine cannot automatically
            detect the separator, but the Python parsing engine can, meaning the
            latter will be used and automatically detect the separator by Python's
            builtin sniffer tool, csv.Sniffer.

        delimiter: `str, default None`
            Alias for sep.

        Attributes
        ----------
        n: lenght of dataset.
        start: start iterator.
        end: end iterator.
        num: current iterator.

        Returns
        -------
        data: Pandas DataFrame, [n_samples, n_classes]
            Dataframe from dataset.

        Examples
        --------
        For usage examples, please see:
        https://jaisenbe58r.github.io/MLearner/user_guide/load/DataLoad/

        """
        data = pd.read_csv(filename, sep=sep, decimal=decimal, **params)
        return cls(data)

    @classmethod
    def load_dataframe(cls, data):
        if not isinstance(data, pd.core.frame.DataFrame):
            raise TypeError("Invalid type {}".format(type(data)))
        return cls(data)

    def reset(self):
        self.num = 0

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.data.iloc[self.num - 1].head()

    def __len__(self):
        return self.n
