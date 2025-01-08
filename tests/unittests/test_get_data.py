# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
from utility_functions.get_data import GetData


class TestGetData(unittest.TestCase):
    def setUp(self):
        """
        Setup for the tests. Initialize GetData with a fixed seed.
        """
        self.seed = 42  # Fixed seed for reproducibility
        self.get_data = GetData(self.seed)

    def test_get_fish_data(self):
        """
        Test the get_fish_data method to ensure it returns a valid filtered DataFrame.
        """
        # Call the method to get the filtered fish data
        filtered_data = self.get_data.get_fish_data()

        # Check the returned data type
        self.assertIsInstance(filtered_data, pd.DataFrame, "The result should be a pandas DataFrame.")

        # Check that the DataFrame has the expected columns
        expected_columns = {"Weight", "Width", "Height"}
        self.assertTrue(expected_columns.issubset(filtered_data.columns), "The DataFrame should have the required columns.")

        # Check that only one length column is present
        length_columns = [col for col in filtered_data.columns if "Length" in col]
        self.assertEqual(len(length_columns), 1, "The DataFrame should have exactly one length column.")

        # Check that the DataFrame is not empty
        self.assertGreater(len(filtered_data), 0, "The DataFrame should not be empty.")

    def test_random_seed_consistency(self):
        """
        Test that using the same seed produces consistent results.
        """
        # Call the method twice with the same seed
        filtered_data_1 = self.get_data.get_fish_data()
        filtered_data_2 = GetData(self.seed).get_fish_data()

        # Check that the results are identical
        pd.testing.assert_frame_equal(filtered_data_1, filtered_data_2, "The results should be identical for the same seed.")


if __name__ == "__main__":
    unittest.main()
