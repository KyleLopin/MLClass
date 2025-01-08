# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
from utility_functions.get_data import load_soybean_data


class TestLoadSoybeanData(unittest.TestCase):

    def test_load_soybean_data(self):
        """Test if the soybean dataset is fetched and returned as a DataFrame."""
        # Call the function
        df = load_soybean_data()
        # Assert that a DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)

        # Check that the DataFrame has data
        self.assertGreater(len(df), 0)

        # Check for specific columns (adjust as per the dataset structure)
        expected_columns = ["PH", "NLP", "GY"]  # Replace with actual column names
        for col in expected_columns:
            self.assertIn(col, df.columns)


if __name__ == '__main__':
    TestLoadSoybeanData.test_load_soybean_data()
