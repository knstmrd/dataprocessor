from os.path import isfile, isdir
from shutil import rmtree
from os import getcwd
import unittest
from src import dataprocessor, removers
import pandas as pd


class DataProcessorTest(unittest.TestCase):
    def setUp(self):
        self.curr_dir = getcwd() + '/'
        self.df = pd.DataFrame({'x': [-2, 0, 2], 'x_p_1': [-1, 1, 3], 'x_t_x': [4, 0, 4], 'nonfeat': ['a', 'b', 'c'],
                                'const': [10, 10, 10]})
        self.dataprocessor = dataprocessor.DataProcessor(self.curr_dir, self.df, non_feature_columns=['nonfeat'])

    def tearDown(self):
        rmtree(self.curr_dir + 'dataprocessor_files/')

    def test_directory_creation(self):
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/correlated'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/unimportant'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/lists'), True)

        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/logs/cv'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/logs/predictions'), True)

        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/settings'), True)

    def test_exclude_nonfeature_columns(self):
        self.assertNotIn('nonfeat', self.dataprocessor.features['all'])
        self.assertIn('x', self.dataprocessor.features['all'])

    def test_removing_features(self):
        self.dataprocessor.add_remover(removers.CorrelatedFeatureRemover, {'correlation_threshold': 0.5})
        self.dataprocessor.add_remover(removers.AlmostConstantFeatureRemover, {'max_count_percent': 80})
        selected, removed = self.dataprocessor.fit_remove(self.df)
        self.assertIn('x', selected)  # we keep the first feature
        self.assertNotIn('x_p_1', selected)  # we don't keep the second feature
        self.assertIn('x_p_1', removed)  # we store the second feature in a separate field
        self.assertNotIn('nonfeat', selected)
        self.assertNotIn('nonfeat', removed)
        self.assertNotIn('const', selected)
        self.assertIn('const', removed)
