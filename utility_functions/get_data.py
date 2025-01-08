# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_soybean_data():
    """
    Fetches the Soybean dataset from the UCIMLRepo and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the soybean dataset.
    """
    # Fetch the dataset from UCIMLRepo
    try:
        dataset = fetch_ucirepo(id=913)  # ID for the Soybean dataset
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        return df["PH", "GY"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch the Soybean dataset: {e}")
