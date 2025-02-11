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
    """
    A utility class for loading and processing datasets for machine learning tasks.

    This class provides methods to:
    - Load a fish dataset, filter it by a randomly selected species and length column,
      and generate prediction datasets based on the selected length.
    - Load the Iris dataset, split it into training and test sets, and prepare prediction datasets.

    Attributes:
        random_gen (random.Random): A random number generator initialized with a seed to ensure reproducibility.

    Methods:
        load_data(num_points=5, test_size=0.05):
            Load and prepare all datasets (Fish and Iris) in one call.

        _get_fish_data(num_points=5):
            Load the fish dataset, filter it by species and length column, and generate prediction datasets.

        _get_iris_data(test_size=0.05):
            Load the Iris dataset, split it into training and test sets, and prepare prediction datasets.
    """
    def __init__(self, seed):
        """
        Initialize the GetData class with a random generator.

        Args:
            seed (int): Seed for the random generator.
        """
        self.random_gen = random.Random(seed)  # Create an isolated random generator

    def load_data(self, num_points=5, test_size=0.05, num_apartments=10, rent_test_size=0.2):
        """
        Load and prepare all necessary datasets (Fish, Iris, and Apartment Rents) in one call.

        Args:
            num_points (int): Number of prediction points to generate for Fish.
            test_size (float): Proportion of the Iris dataset to include in the test split.
            num_apartments (int): Number of rows to generate for the Apartment Rents dataset.
            rent_test_size (float): Proportion of the Apartment Rents dataset to include in the test split.

        Returns:
            dict: A dictionary containing the prepared datasets:
                - 'fish_data': Filtered fish datasets with selected species and columns.
                - 'fish_prediction': Prediction datasets for fish.
                - 'iris_train': Training datasets for Iris.
                - 'iris_prediction': Prediction (test) datasets for Iris.
                - 'apartment_train': Training datasets for apartment rents.
                - 'apartment_test': Test datasets for apartment rents.
        """
        # Call private methods to load individual datasets
        fish_data, fish_prediction = self._get_fish_data(num_points)
        iris_train, iris_prediction = self._get_iris_data(test_size)
        apartment_train, apartment_test = self._get_apartment_rents(num_apartments, rent_test_size)

        # Return all datasets in a dictionary
        return {
            "fish_data": fish_data,
            "fish_prediction": fish_prediction,
            "iris_train": iris_train,
            "iris_prediction": iris_prediction,
            "apartment_train": apartment_train,
            "apartment_prediction": apartment_test,
        }

    def _get_fish_data(self, num_points=5):
        """
        Randomly select one species and one Length column (keeping Width, Height, and Weight)
        from the fish dataset, return a filtered DataFrame, and generate prediction datasets.

        Args:
            num_points (int): Number of prediction points to generate.

        Returns:
            dict: A dictionary containing:
                - "fish_data": Filtered DataFrame with the selected species and columns.
                - "fish_prediction": DataFrame with generated prediction datasets.
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
        selected_length = self.random_gen.choice(length_columns)

        # Create a filtered DataFrame with the selected Length column and other required columns
        fish_data = df_species[["Weight", selected_length, "Width", "Height"]].copy()

        # Get the range of the selected length column
        length_values = fish_data[selected_length]

        # Generate uniformly distributed random numbers for the length column
        generated_lengths = [
            round(self.random_gen.uniform(length_values.min(), length_values.max()), 1) for _ in range(num_points)
        ]

        # Create a DataFrame with the generated lengths
        prediction_df = pd.DataFrame({
            selected_length: generated_lengths
        })

        # Return both fish datasets and prediction datasets
        return fish_data.reset_index(drop=True), prediction_df

    def _get_iris_data(self, test_size=0.05):
        """
        Load the Iris dataset, split it into training and test sets, and prepare prediction datasets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: Training datasets with the target column.
                - pd.DataFrame: Prediction datasets (test set without the target column).
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
        iris_train = train.reset_index(drop=True)
        iris_test = test.reset_index(drop=True)

        # Create the prediction set by removing the target column from the test set
        iris_prediction = iris_test.drop(columns=["target"])

        # Return both training datasets and prediction datasets
        return iris_train, iris_prediction

    def _get_apartment_rents(self, num_apartments=10, test_size=0.2):
        """
        Generate synthetic datasets for apartment rents with increments of 100 baht and split into training and test sets.

        Args:
            num_apartments (int): Total number of rows to generate.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: A tuple containing:
                - train_data (pd.DataFrame): Training dataset.
                - test_data (pd.DataFrame): Test dataset.
        """
        # Generate random apartment sizes
        square_meters = [self.random_gen.randint(30, 80) for _ in range(num_apartments)]

        # Define a base rent multiplier (in increments of 100)
        base_rate = self.random_gen.choice([2, 3])  # Base rate is either 200 or 300 baht
        coefficient = self.random_gen.randrange(1, 3)  # Random coefficient for rent

        # Calculate rents with noise
        rents = [
            (base_rate + coefficient * sm + self.random_gen.randint(-1, 5)) * 100
            for sm in square_meters
        ]

        # Create the full dataset
        apartment_rents = pd.DataFrame({
            "square_meters": square_meters,
            "rent (baht)": rents,
        })

        # Split into training and test sets
        train_data, test_data = train_test_split(
            apartment_rents, test_size=test_size, random_state=self.random_gen.randint(0, 2 ** 32 - 1)
        )

        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


# Example Usage
if __name__ == '__main__':
    getdata = GetData(43)
    datasets = getdata.load_data()

    print("Fish Data:")
    print(datasets["fish_data"])

    print("\nFish Prediction Data:")
    print(datasets["fish_prediction"])

    print("\nIris Training Data:")
    print(datasets["iris_train"])

    print("\nIris Prediction Data:")
    print(datasets["iris_prediction"])

    print("\nApartment Training Data:")
    print(datasets["apartment_train"])

    print("\nApartment Prediction Data:")
    print(datasets["apartment_prediction"])

    print(datasets.keys())
