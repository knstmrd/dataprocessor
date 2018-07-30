import pandas as pd
import numpy as np
import unittest
from src import transforms
from src import removers


class DataTransformsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'x': [0, 1e7], 'y': [0, 9e4], 'z': [1e-5, 1e2]})
        self.df2 = pd.DataFrame({'x': [-2, 0, 2], 'x_p_1': [-1, 1, 3], 'x_t_x': [4, 0, 4], 'nonfeat': ['a', 'b', 'c']})
        self.df3 = pd.DataFrame({'x': [0, 0, 0, 0, 1], 'y': [0, 0, 0, 1, 1], 'z': [100, 100, 100, 100, 100],
                                 'txt': ['a', 'b', 'c', 'd', 'e']})
        self.df4 = pd.DataFrame({'x': [-2, 0, 2], 'x_p_1': [-100, 21, 3], 'x_t_x': [4, 0, 4]})

    def test_log_scaling_pandas(self):
        df_copy = self.df.copy()
        logtransform = transforms.LogTransformer()
        logtransform.fit(df_copy)
        scaled_df = logtransform.transform(df_copy)
        self.assertIn('x', logtransform.column_names_)
        self.assertIn('z', logtransform.column_names_)
        self.assertNotIn('y', logtransform.column_names_)

        self.assertLess(scaled_df['x'].max(), 20)
        self.assertEqual(scaled_df['z'][0], 0)
        self.assertEqual(scaled_df['y'][1], 9e4)

    def test_find_correlated_features(self):
        corr_remover = removers.CorrelatedFeatureRemover(0.5)
        selected, correlated = corr_remover.fit(self.df2, ['x', 'x_p_1', 'x_t_x'])
        self.assertIn('x', selected)  # we keep the first feature
        self.assertNotIn('x_p_1', selected)  # we don't keep the second feature
        self.assertIn('x_p_1', correlated)  # we store the second feature in a separate field
        self.assertNotIn('nonfeat', selected)
        self.assertNotIn('nonfeat', correlated)

    def test_find_almostconst_features(self):
        almostconst_remover = removers.AlmostConstantFeatureRemover(max_count_percent=75)
        selected, removed = almostconst_remover.fit(self.df3, self.df3.columns)
        self.assertIn('x', removed)
        self.assertNotIn('x', selected)
        self.assertIn('z', removed)
        self.assertNotIn('y', removed)
        self.assertIn('y', selected)
        self.assertNotIn('txt', removed)

    def test_correlated_features_persistence(self):
        corr_remover = removers.CorrelatedFeatureRemover(0.5, write_to_file='test.csv')
        selected, correlated = corr_remover.fit(self.df2, ['x', 'x_p_1', 'x_t_x'])
        corr_remover2 = removers.CorrelatedFeatureRemover(0.5, load_from_file='test.csv')

        # the features are different, but since we read from file, it should remove based on the old dataframe
        selected, correlated = corr_remover2.fit(self.df4, ['x', 'x_p_1', 'x_t_x'])
        self.assertIn('x', selected)  # we keep the first feature
        self.assertNotIn('x_p_1', selected)  # we don't keep the second feature
        self.assertIn('x_p_1', correlated)  # we store the second feature in a separate field

