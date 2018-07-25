import pathlib
from datetime import datetime
from itertools import chain
import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self, path: str, df, non_feature_columns=None, fname_prefix='', verbose=True):
        if not path.endswith('/'):
            path += '/'
        self.base_path = path
        self.fname = fname_prefix + str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
        self.removers = []
        self.remover_params = []
        self.transforms = []
        self.transform_params = []

        self.features = {'all': [col for col in df.columns if col not in non_feature_columns],
                         'selected': [],
                         'removed': []
                        }

        self.non_feature_columns = non_feature_columns  # stuff like label, filename, etc.

        self.verbose = verbose

        # store lists of features to be removed
        pathlib.Path(path + 'dataprocessor_files/features/correlated').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path + 'dataprocessor_files/features/unimportant').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path + 'dataprocessor_files/features/lists').mkdir(parents=True, exist_ok=True)

        # store logs for cv and predictions
        pathlib.Path(path + 'dataprocessor_files/logs/cv').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path + 'dataprocessor_files/logs/predictions').mkdir(parents=True, exist_ok=True)

        pathlib.Path(path + 'dataprocessor_files/settings').mkdir(parents=True, exist_ok=True)

    def return_features_list(self, use_features: str='selected'):
        """
        Return list of features - return either all features, or selected features, or removed features,
        or some union of these
        :param use_features:
        :return:
        """
        return self.features[use_features]

    def add_remover(self, remover, remover_params):
        self.removers.append(remover(**remover_params))
        self.remover_params.append(remover_params)
        return self

    def fit_remove(self, df):
        current_features_to_use = self.return_features_list('all')
        features_removed = []
        for remover in self.removers:
            current_features_to_use, to_be_removed = remover.fit(df, current_features_to_use)
            features_removed += to_be_removed
        features_removed = list(set(features_removed))
        self.features['removed'] = features_removed
        self.features['selected'] = current_features_to_use
        return self.features['selected'], self.features['removed']

    def add_transform(self, transform, transform_params):
        self.transforms.append(transform(**transform_params))
        self.transform_params.append(transform_params)
        return self

    def fit_transform(self, df, use_features: str='selected'):
        """
        Fit and apply transforms to a dataframe
        :param df:
        :param transform:
        :param use_features:
        :return:
        """
        features_to_use = self.return_features_list(use_features)

        for transform, transform_params in zip(self.transforms, self.transform_params):
            df[features_to_use] = transform.fit_transform(df[features_to_use])

    def transform(self, df, use_features: str='selected'):
        """
        Apply transforms to a dataframe
        :param df:
        :param use_features:
        :return:
        """
        features_to_use = self.return_features_list(use_features)

        for transform, transform_params in zip(self.transforms, self.transform_params):
            df[features_to_use] = transform.transform(df[features_to_use], **transform_params)
