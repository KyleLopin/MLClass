# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Unittest Module for Testing the GetData Class

This module tests the functionality of the GetData class, which provides utilities
for loading and preparing Fish and Iris datasets. The tests ensure that the data
is loaded correctly, random selections are consistent with a fixed seed, and that
predictions and training data are structured as expected.

Tests Included:
- Loading and filtering the Fish dataset by species and length columns.
- Generating prediction data for the Fish dataset.
- Splitting the Iris dataset into training and test sets.
- Ensuring random seed consistency across instances.
- Verifying the integration of the `load_data` method.

Usage:
    Run this module with Python's unittest framework:
        python -m unittest test_getdata.py

Dependencies:
    - unittest (standard library)
    - pandas
    - GetData class from your_module (replace with actual module name)
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
from utility_functions.get_data import GetData


class TestGetData(unittest.TestCase):
    """
    Unit Tests for the GetData Class

    This test suite verifies the functionality of the GetData class, including:
    - Loading the Fish dataset and generating prediction data.
    - Loading the Iris dataset, splitting it into training and test sets, and preparing predictions.
    - Ensuring consistency of random outputs with a fixed seed.
    - Testing the integration of the `load_data` method for both datasets.

    Methods:
        - setUp: Initializes a GetData instance with a fixed seed for reproducibility.
        - test_get_fish_data: Tests the filtering and prediction generation for the Fish dataset.
        - test_get_iris_data: Tests the splitting and preparation of the Iris dataset.
        - test_load_data: Tests the integration of all dataset operations via `load_data`.
        - test_random_seed_consistency: Ensures that a fixed seed produces consistent outputs.
    """
    def setUp(self):
        """
        Set up a GetData instance with a fixed seed before each test.
        """
        self.seed = 42
        self.getdata = GetData(seed=self.seed)

    def test_get_fish_data(self):
        """
        Test the _get_fish_data method.
        """
        # Call the method
        fish_data, fish_prediction = self.getdata._get_fish_data(num_points=3)

        # Verify the fish data
        self.assertIn("Weight", fish_data.columns)
        self.assertIn("Width", fish_data.columns)
        self.assertIn("Height", fish_data.columns)

        # Verify the prediction data
        self.assertEqual(len(fish_prediction), 3)
        self.assertIn(self.getdata.random_gen.choice(["Length1", "Length2"]), fish_prediction.columns)

    def test_get_iris_data(self):
        """
        Test the _get_iris_data method.
        """
        # Call the method
        iris_train, iris_prediction = self.getdata._get_iris_data(test_size=0.1)

        # Verify the training data
        self.assertIn("target", iris_train.columns)
        self.assertGreater(len(iris_train), 0)

        # Verify the prediction data
        self.assertNotIn("target", iris_prediction.columns)
        self.assertGreater(len(iris_prediction), 0)

    def test_load_data(self):
        """
        Test the load_data method.
        """
        # Call the load_data method
        datasets = self.getdata.load_data()

        # Verify fish data
        self.assertIn("fish_data", datasets)
        self.assertIn("fish_prediction", datasets)

        # Verify iris data
        self.assertIn("iris_train", datasets)
        self.assertIn("iris_prediction", datasets)

        # Verify prediction data lengths
        self.assertGreater(len(datasets["fish_prediction"]), 0)
        self.assertGreater(len(datasets["iris_train"]), 0)

    def test_random_seed_consistency(self):
        """
        Test that the random generator produces consistent results with the same seed.
        """
        seed = 123
        getdata1 = GetData(seed)
        getdata2 = GetData(seed)

        # Verify that the same random fish data is generated
        fish_data1, fish_prediction1 = getdata1._get_fish_data()
        fish_data2, fish_prediction2 = getdata2._get_fish_data()
        pd.testing.assert_frame_equal(fish_data1, fish_data2)
        pd.testing.assert_frame_equal(fish_prediction1, fish_prediction2)

        # Verify that the same iris data is generated
        iris_train1, iris_prediction1 = getdata1._get_iris_data()
        iris_train2, iris_prediction2 = getdata2._get_iris_data()
        pd.testing.assert_frame_equal(iris_train1, iris_train2)
        pd.testing.assert_frame_equal(iris_prediction1, iris_prediction2)



if __name__ == "__main__":
    unittest.main()
