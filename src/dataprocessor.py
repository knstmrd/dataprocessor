import pathlib
from datetime import datetime
import json


class DataProcessor:
    def __init__(self, path: str, df, non_feature_columns=None, fname_prefix='', verbose=True):
        if not path.endswith('/'):
            path += '/'
        self.base_path = path
        self.removers = []
        self.fname_prefix = fname_prefix
        self.fname = self.fname_prefix + str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
        self.remover_params = []
        self.transforms = []
        self.transform_params = []

        self.saved = False

        self.features = {'all': [col for col in df.columns if col not in non_feature_columns],
                         'selected': [],
                         'removed': []
                        }

        self.non_feature_columns = non_feature_columns  # stuff like label, filename, etc.

        self.verbose = verbose

        if pathlib.Path(path + 'dataprocessor_files/settings/current_settings.log').exists():
            with open(self.base_path + 'dataprocessor_files/settings/current_settings.log', 'r') as f:
                self.settings = json.load(f)
            print('Previous settings file found')
            with open(path + self.settings['features removed list'], 'r') as f:
                self.features['removed'] = [l.replace('\n', '') for l in f if l != '\n']
            print('List of removed features contains {} elements'.format(len(self.features['removed'])))
            with open(path + self.settings['features selected list'], 'r') as f:
                self.features['selected'] = [l.replace('\n', '') for l in f if l != '\n']
            print('List of selected features contains {} elements'.format(len(self.features['selected'])))
        else:
            print('No previous settings file found')
            self.settings = {}

        # store lists of features to be removed
        pathlib.Path(path + 'dataprocessor_files/features/removed').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path + 'dataprocessor_files/features/selected').mkdir(parents=True, exist_ok=True)

        # store logs for cv and predictions
        pathlib.Path(path + 'dataprocessor_files/output/cv').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path + 'dataprocessor_files/output/predictions').mkdir(parents=True, exist_ok=True)

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

    def cv(self, df, predictor, scorers, predict_proba=False, df_test=None, use_features: str='selected'):

        mean_score = [0.0] * len(scorers)
        std_score = [0.0] * len(scorers)
        if not self.saved:
            print('Dataprocessor settings not saved, will save now')
            self.save()

        with open(self.base_path + 'dataprocessor_files/cv/cv.log', 'a') as f:

            for (scorer, scorer_name) in scorers:
                f.write('Mean(score): {}\nStd(score): {}\n'.format(mean_score, std_score))
                f.write('Scorer={}, mean(score): {}\nScorer={}, std(score): {}\n'.format(scorer_name, mean_score,
                                                                                         scorer_name, std_score))
            f.write('Predictor: {}\nPredict_proba: {}\n'.format(str(predictor), str(predict_proba)))
            f.write('Features used: {}\n'.format(use_features))
            f.write('Selected features list: {}\n'.format(self.base_path + 'dataprocessor_files/features/selected/'
                                                          + self.fname))
            f.write('Removed features list: {}\n'.format(self.base_path + 'dataprocessor_files/features/removed/'
                                                         + self.fname))

            f.write('\n\n\n')
        # Write mean, std (score); names of feature lists, list of feature removers and transforms and their settings
        # also can run on df_test and predict there
        pass

    def save(self):
        self.fname = self.fname_prefix + str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
        with open(self.base_path + 'dataprocessor_files/features/removed/' + self.fname, 'w') as f:
            for feature in self.features['removed']:
                f.write('{}\n'.format(feature))

        with open(self.base_path + 'dataprocessor_files/features/selected/' + self.fname, 'w') as f:
            for feature in self.features['selected']:
                f.write('{}\n'.format(feature))

        self.settings = {'features removed list': 'dataprocessor_files/features/removed/' + self.fname,
                         'features selected list': 'dataprocessor_files/features/selected/' + self.fname,
                         'removers': [str(remover) for remover in self.removers],
                         'remover_params': [str(remover_params) for remover_params in self.remover_params],
                         'fname': self.fname}

        # move old settings
        if pathlib.Path(self.base_path + 'dataprocessor_files/settings/current_settings.log').exists():
            with open(self.base_path + 'dataprocessor_files/settings/current_settings.log', 'r') as f:
                old_settings = json.load(f)

            with open(self.base_path
                      + 'dataprocessor_files/settings/old_settings_{}.log'.format(old_settings['fname']), 'w') as f:
                json.dump(old_settings, f)

        with open(self.base_path + 'dataprocessor_files/settings/current_settings.log', 'w') as f:
            json.dump(self.settings, f)
        self.saved = True

    def write_features(self, df, df_format):
        # write selected and transformed features to a dataframe
        # and add that to settings
        pass
