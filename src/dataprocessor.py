import pathlib
from datetime import datetime
from itertools import chain


class DataProcessor:
    def __init__(self, path: str, non_feature_columns=None):
        if not path.endswith('/'):
            path += '/'
        self.base_path = path
        self.fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
        self.transforms = []

        self.features = {'all': [],
                         'selected': [],
                         'correlated': [],
                         'unimportant': []}

        self.correlation_threshold = 1.0
        self.importance_threshold = -1.0
        self.non_feature_columns = non_feature_columns  # stuff like label, filename, etc.

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
        Return list of features - return either all features, or selected features, correlated, unimportant,
        or some union of these
        :param use_features:
        :return:
        """
        if use_features in ('selected', 'correlated', 'unimportant', 'all'):
            return self.features[use_features]
        elif '_' in use_features:
            split = use_features.split('_')
            output = [self.features[x] for x in split]
            output = list(set(list(chain(*output))))
            return output

    def fit_transform(self, df, transform, use_features: str='selected'):
        """
        Fit and apply transforms to a dataframe
        :param df:
        :param transform:
        :param use_features:
        :return:
        """
        features_to_use = self.return_features_list(use_features)
        df[features_to_use] = transform.fit_transform(df[features_to_use])
        self.transforms.append(transform)

    def transform(self, df, use_features: str='selected'):
        """
        Apply transforms to a dataframe
        :param df:
        :param use_features:
        :return:
        """
        features_to_use = self.return_features_list(use_features)
        for transform in self.transforms:
            df[features_to_use] = transform.fit_transform(df[features_to_use])
