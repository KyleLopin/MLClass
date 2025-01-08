# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import random

# installed libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class GetData:
    def __init__(self, seed):
        """
        Initialize the GetData class with a random generator.

        Args:
            seed (int): Seed for the random generator.
        """
        self.random_gen = random.Random(seed)  # Create an isolated random generator
        self.fish_data = None  # To store the loaded fish DataFrame
        self.selected_length = None  # To store the selected length column name

    def get_fish_data(self):
        """
        Randomly select one species and one Length column (keeping Width, Height, and Weight)
        from the fish dataset and return a filtered DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame with the selected species and columns.

        """
        # Load the fish dataset
        df = pd.read_csv("hf://datasets/scikit-learn/Fish/Fish.csv")

        # Randomly select one species
        unique_species = df["Species"].unique()
        selected_species = self.random_gen.choice(unique_species)

        # Filter the DataFrame for the selected species
        df_species = df[df["Species"] == selected_species]

        # Randomly select one of the Length columns
        length_columns = [col for col in df.columns if "Length" in col]
        self.selected_length = self.random_gen.choice(length_columns)

        # Create a filtered DataFrame with the selected Length column and other required columns
        self.fish_data = df_species[["Weight", self.selected_length, "Width", "Height"]].copy()

        return self.fish_data.reset_index(drop=True)

    def get_prediction_data(self, num_points=5):
        """
        Generate prediction data by sampling numbers normally distributed over the range of
        the selected length column in the bound fish dataset.

        Args:
            num_points (int): Number of prediction points to generate.

        Returns:
            pd.DataFrame: DataFrame with the generated prediction points.
        """
        if self.fish_data is None or self.selected_length is None:
            raise ValueError("Fish data has not been loaded. Call 'get_fish_data' first.")

        # Get the range of the selected length column
        length_values = self.fish_data[self.selected_length]
        mean_length = length_values.mean()
        std_length = length_values.std()

        # Generate normally distributed random numbers for the length column
        generated_lengths = self.random_gen.gauss(mean_length, std_length)  # Single point
        generated_lengths = [
            self.random_gen.gauss(mean_length, std_length) for _ in range(num_points)
        ]

        # Create a DataFrame with the generated lengths
        prediction_df = pd.DataFrame({
            self.selected_length: generated_lengths
        })
        return prediction_df

    def get_iris_data(self, test_size=0.05):
        """
        Load the Iris dataset and split it into training and test sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            pd.DataFrame: Training data.
        """
        # Load the Iris dataset
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['target'] = iris.target

        # Split the dataset into training and test sets
        train, test = train_test_split(
            data, test_size=test_size, random_state=self.random_gen.randint(0, 2 ** 32 - 1)
        )

        # Store the splits
        self.iris_train = train.reset_index(drop=True)
        self.iris_test = test.reset_index(drop=True)

        return self.iris_train

    def get_iris_prediction(self):
        """
        Get the test split of the Iris dataset, excluding the "target" column.

        Returns:
            pd.DataFrame: Test data with the "target" column removed.
        """
        if self.iris_test is None:
            raise ValueError("Iris data has not been loaded. Call 'get_iris_data' first.")

        return self.iris_test.drop(columns=["target"])


if __name__ == '__main__':
    getdata = GetData(43)
    print(getdata.get_fish_data())
    print(getdata.get_prediction_data())
    print(getdata.get_iris_data())
    print('=====')
    print(getdata.get_iris_prediction())
