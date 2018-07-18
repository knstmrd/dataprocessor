import pandas as pd
import numpy as np
import unittest
from src import transforms
from src import removers


class DataTransformsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'x': [0, 1e7], 'y': [0, 9e4], 'z': [1e-5, 1e2]})
        self.df2 = pd.DataFrame({'x': [-2, 0, 2], 'x_p_1': [-1, 1, 3], 'x_t_x': [4, 0, 4], 'nonfeat': ['a', 'b', 'c']})

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
