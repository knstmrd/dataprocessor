import pandas as pd
import numpy as np


class BaseFeatureRemover:
    pass


class PredictorBasedFeatureRemover:
    pass


class CorrelatedFeatureRemover:
    def __init__(self, correlation_threshold, verbose=True, force_recompute=False):
        self.correlation_threshold = correlation_threshold
        self.columns_to_remove = []
        self.columns_to_leave = []
        self.fitted = False
        self.verbose = verbose
        self.force_recompute = force_recompute

    def fit(self, df, feature_columns):
        corr = df[feature_columns].corr()
        corr = corr.abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        self.columns_to_remove = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        self.columns_to_leave = [x for x in feature_columns if x not in self.columns_to_remove]
        self.fitted = True

        if self.verbose:
            print(str(len(self.columns_to_remove))
                  + ' features found with a correlation higher than ' + str(self.correlation_threshold))

        return self.columns_to_leave, self.columns_to_remove
