import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .exceptions import DataProcessorError
from sklearn.exceptions import NotFittedError


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    If the ratio of max(abs(X[:, col]))/min(abs(X[:, col])) exceeds a certain threshold,
    replace the values in the column with a logarithm: X[i, col] = np.log(1 + X[i, col] - min(X[:, col]))
    """
    def __init__(self, threshold=1e5):
        self.fitted = False
        self.threshold = threshold

    def __str__(self):
        return 'LogTransformer(threshold={})'.format(self.threshold)

    def _reset(self):
        if hasattr(self, 'columns_'):
            del self.columns_
            del self.column_names_
            del self.min_vals_
            self.fitted = False

    def fit(self, X):
        self._reset()
        self.columns_ = []
        self.column_names_ = []
        self.min_vals_ = []

        if type(X) == pd.DataFrame:
            Y = X.values
        else:
            Y = X

        min_vals = np.min(Y, axis=0)
        max_vals = np.max(Y, axis=0)

        zero_min = np.where(min_vals == 0)[0]
        non_zero_min = np.where(min_vals != 0)[0]
        large_max = np.where(max_vals > self.threshold)[0]

        large_max_zero_min = [x for x in zero_min if x in large_max]

        if large_max_zero_min:
            for i in large_max_zero_min:
                self.columns_.append(i)
                self.min_vals_.append(0)

        for i in non_zero_min:
            if abs(max_vals[i] / min_vals[i]) > self.threshold:
                self.columns_.append(i)
                self.min_vals_.append(min_vals[i])

        if type(X) == pd.DataFrame:
            self.column_names_ = [X.columns[i] for i in self.columns_]

        self.fitted = True

        return self

    def transform(self, X):
        if self.fitted:
            if type(X) == pd.DataFrame:
                Y = X.values
            else:
                Y = X

            for i, col in enumerate(self.columns_):
                Y[:, col] = np.log1p(Y[:, col] - self.min_vals_[i])

            return X
        else:
            raise NotFittedError('This LogTransformer has not been fitted yet')
