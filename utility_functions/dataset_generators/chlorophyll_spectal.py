# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Get spectral data for chlorophyll measurments, now it is just collected data from my article:
"Evaluation of Low-Cost Multi-Spectral Sensors for Measuring
Chlorophyll Levels Across Diverse Leaf Types"
"""

__author__ = "Kyle Vitautas Lopin"


# standard libraries
from pathlib import Path
import random  # for testing in __main__
import sys

# installed libraries
import pandas as pd


IN_COLAB = "google.colab" in sys.modules


def get_spectral_data(random_gen: random.Random):
    """

    Args:
        random_gen (random.Random): random generator to ensure consistent data retrieval

    Returns:
        pd.DataFrame:

    """
    sensor = random_gen.choice(["as7262", "as7263", "as7265x"])
    leaf = random.choice(["banana", "rice", "mango", "rice", "sugarcane"])
    if IN_COLAB:
        df_sensor = pd.read_csv(f"/content/MLClass/datasets/{sensor}_leaves.csv")
    else:  # home computer for testing
        file_path = Path(__file__).resolve().parents[2] / "datasets"
        df_sensor = pd.read_csv(file_path / f"{sensor}_leaves.csv")

    df_sensor = df_sensor[df_sensor["leaf"] == leaf]

    return df_sensor


if __name__ == '__main__':
    df = get_spectral_data(random.Random())
