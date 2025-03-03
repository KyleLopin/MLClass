# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import importlib
import random

# installed libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# local files
import get_fish_data


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

    def load_data(self, dataset_loader: str, **kwargs):
        """
        Load and prepare the dataset based on the specified dataset type.

        Args:
            dataset_type (str): The dataset to load (e.g., "fish", "iris", "weather").
            **kwargs: Additional arguments for dataset processing (e.g., date, test_size).

        Returns:
            tuple: Processed datasets .
        """
        if "fish" in dataset_loader:
            return get_fish_data.get_fish_data(self.random_gen, dataset_loader)
        else:
            raise ValueError(f"Dataset '{dataset_loader}' is not supported.")

    def _get_dataset_loader(self):
        """
        Dynamically load the appropriate dataset module based on `dataset_type`.

        Returns:
            function: A function that loads and processes the dataset.
        """
        try:
            # Import dataset processing module dynamically
            dataset_module = importlib.import_module(
                f"utility_functions.get_{self.dataset_type}_data")
            return dataset_module.load_dataset  # Ensure each module has a `load_dataset()` function
        except ModuleNotFoundError:
            print(f"⚠️ Warning: Dataset module 'get_{self.dataset_type}_data' not found.")
            return None

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
        data['target'] = data['target'].map({i: name for i, name in enumerate(iris.target_names)})

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


def check_loaded():
    print("\n\n\n\n================Loaded properly !!! ================ \n")


# Example Usage
if __name__ == '__main__':
    getdata = GetData(43)
    dataset = getdata.load_data("fish syn", num_points=2)
    print(dataset)

    dataset = getdata.load_data("fish coeff")
    print(dataset)
    dataset = getdata.load_data("fish cost")

    print(dataset)
    #
    # print("Fish Data:")
    # print(datasets["fish_data"])
    #
    # print("\nFish Prediction Data:")
    # print(datasets["fish_prediction"])
    #
    # print("\nIris Training Data:")
    # print(datasets["iris_train"])
    #
    # print("\nIris Prediction Data:")
    # print(datasets["iris_prediction"])
    #
    # print("\nApartment Training Data:")
    # print(datasets["apartment_train"])
    #
    # print("\nApartment Prediction Data:")
    # print(datasets["apartment_prediction"])
    #
    # print(datasets.keys())
