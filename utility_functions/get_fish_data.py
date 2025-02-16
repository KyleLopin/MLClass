# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import random

# installed libraries
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_fish_data(random_gen: random.Random, dataset: str,
                 num_points: int = 3, **kwargs):
    """
    Load the fish dataset.

    - If `dataset == "fish coeff"`, selects a random species and computes its weight coefficient.
    - If `dataset == "fish cost"`, selects a random species, picks random lengths,
      predicts weight, calculates cost, and returns the price.

    Args:
        random_gen (random.Random): Seeded random generator for reproducibility.
        dataset (str): The dataset type, either "fish coeff" or "fish cost".
        num_points (int): Number of random length values for "fish cost" (default=2).
        **kwargs: Additional parameters.

    Returns:
        - (species_name, weight_coefficient) if `dataset="fish coeff"`
        - (species_name, random_lengths, price_per_100g, total_cost) if `dataset="fish cost"`
    """
    df = pd.read_csv("hf://datasets/scikit-learn/Fish/Fish.csv")
    fish_types = ["Bream", "Pike", "Smelt", "Parkki", "Whitefish"]
    selected_fish = random_gen.choice(fish_types)

    # Handle "fish coeff" case
    if dataset == "fish coeff":
        df_species = df[df["Species"] == selected_fish]

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(df_species[["Length3"]], df_species["Weight"])
        weight_coefficient =round(model.coef_[0], 2)

        return selected_fish, weight_coefficient

    # Handle "fish cost" case
    elif dataset == "fish cost":
        df_species = df[df["Species"] == selected_fish]

        # Get min/max length for this species
        min_length, max_length = df_species["Length3"].min(), df_species["Length3"].max()

        # Randomly select `num_points` lengths within this range
        random_lengths = [round(random_gen.uniform(min_length, max_length), 1) for _ in range(num_points)]

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(df_species[["Length3"]], df_species["Weight"])

        # Convert `random_lengths` to a DataFrame with correct column name
        random_lengths_df = pd.DataFrame(random_lengths, columns=["Length3"])

        # Predict weights
        predicted_weights = model.predict(random_lengths_df)

        # Select a random price per 100g (100-300 Baht)
        price_per_100g = random_gen.randint(10, 30) * 10

        # Compute total cost (convert weight to grams, multiply by price per 100g)
        total_cost = sum(predicted_weights) * (price_per_100g / 100)

        return selected_fish, random_lengths, price_per_100g, total_cost

    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset}. Choose 'fish coeff' or 'fish cost'.")

