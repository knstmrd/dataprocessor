from os.path import isfile, isdir
from shutil import rmtree
from os import getcwd
import unittest
from src import dataprocessor


class DataProcessorTest(unittest.TestCase):
    def setUp(self):
        self.curr_dir = getcwd() + '/'
        self.dataprocessor = dataprocessor.DataProcessor(self.curr_dir)

    def tearDown(self):
        rmtree(self.curr_dir + 'dataprocessor_files/')

    def test_directory_creation(self):
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/correlated'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/unimportant'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/features/lists'), True)

        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/logs/cv'), True)
        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/logs/predictions'), True)

        self.assertEqual(isdir(self.curr_dir + 'dataprocessor_files/settings'), True)

