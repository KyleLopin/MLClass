# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import random

# installed libraries
import pandas as pd


class GetData:
    def __init__(self, seed):
        """
        Initialize the GetData class with a random generator.

        Args:
            seed (int): Seed for the random generator.
        """
        self.random_gen = random.Random(seed)  # Create an isolated random generator

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
        selected_length = self.random_gen.choice(length_columns)

        # Create a filtered DataFrame with the selected Length column and other required columns
        filtered_df = df_species[["Weight", selected_length, "Width", "Height"]].copy()

        return filtered_df.reset_index(drop=True)


if __name__ == '__main__':
    getdata = GetData(43)
    print(getdata.get_fish_data())
