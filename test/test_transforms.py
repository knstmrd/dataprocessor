import pandas as pd
import numpy as np
import unittest
from src import transforms


class DataProcessorTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'x': [0, 1e7], 'y': [0, 9e4], 'z': [1e-5, 1e2]})

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
