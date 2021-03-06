import pandas as pd
import numpy as np
from .exceptions import DataProcessorError


class BaseFeatureRemover:
    pass


class PredictorBasedFeatureRemover:
    pass


class CorrelatedFeatureRemover:
    def __init__(self, correlation_threshold=0.9, verbose=True, force_recompute=False, write_to_file=False,
                 load_from_file=False):
        """

        :param correlation_threshold:
        :param verbose:
        :param force_recompute:
        :param write_to_file: if is a string (filename), write correlation matrix to file
        :param load_from_file: if is a string (filename), load correlation matrix from file
        """
        self.correlation_threshold = correlation_threshold
        self.columns_to_remove = []
        self.columns_to_leave = []
        self.fitted = False
        self.verbose = verbose
        self.force_recompute = force_recompute
        self.persistent = True
        self.write_to_file = write_to_file
        self.load_from_file = load_from_file

    def __str__(self):
        return 'CorrelatedFeatureRemover(correlation_threshold={})'.format(self.correlation_threshold)

    def fit(self, df, feature_columns):
        if self.load_from_file:
            corr = pd.read_csv(self.load_from_file, index_col=0)
            corr = corr.abs()

            for col in feature_columns:
                if col not in corr.columns:
                    raise DataProcessorError("Column '{}' not found in correlation file".format(col))
        else:
            corr = df[feature_columns].corr()
            if self.write_to_file:
                corr.to_csv(self.write_to_file)
            corr = corr.abs()

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        self.columns_to_remove = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        self.columns_to_leave = [x for x in feature_columns if x not in self.columns_to_remove]
        self.fitted = True

        if self.verbose:
            print(str(len(self.columns_to_remove))
                  + ' features found with a correlation higher than ' + str(self.correlation_threshold))

        return self.columns_to_leave, self.columns_to_remove


class AlmostConstantFeatureRemover:
    def __init__(self, max_count_percent=90, verbose=True, force_recompute=False):
        """
        If a column has a single value that makes up more than max_count_percent of the values, remove it
        :param max_count_percent:
        :param verbose:
        :param force_recompute:
        """
        self.max_count_percent = max_count_percent
        self.columns_to_remove = []
        self.columns_to_leave = []
        self.fitted = False
        self.verbose = verbose
        self.force_recompute = force_recompute
        self.persistent = False

    def __str__(self):
        return 'AlmostConstantFeatureRemover(max_count_percent={})'.format(self.max_count_percent)

    def fit(self, df, feature_columns):
        len_df = len(df)

        for col in feature_columns:
            count = df[col].value_counts().values[0]
            if 100 * count / len_df > self.max_count_percent:
                self.columns_to_remove.append(col)

        self.columns_to_leave = [x for x in feature_columns if x not in self.columns_to_remove]
        self.fitted = True

        if self.verbose:
            print(str(len(self.columns_to_remove))
                  + ' features found with a relative count percentage higher than ' + str(self.max_count_percent))

        return self.columns_to_leave, self.columns_to_remove
