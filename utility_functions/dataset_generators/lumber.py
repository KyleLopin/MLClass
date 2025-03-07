# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import random  # for testing in __main__

# installed libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

# Define tree species and their real-world scaling factors
thai_tree_species = ["Teak", "Siamese Rosewood", "Bullet Wood", "Red Cedar", "Pine"]
scaling_factors = {
    "Teak": {"diameter": (20, 70), "length": (6, 25), "taper": (0.5, 1.5), "bark": (1.0, 2.5), "density": 0.70},
    "Siamese Rosewood": {"diameter": (60, 100), "length": (20, 30), "taper": (0.5, 1.0), "bark": (1.5, 2.5), "density": 0.90},
    "Bullet Wood": {"diameter": (20, 60), "length": (10, 20), "taper": (0.4, 0.9), "bark": (0.8, 2.0), "density": 0.80},
    "Red Cedar": {"diameter": (30, 100), "length": (20, 40), "taper": (0.8, 1.3), "bark": (1.2, 2.8), "density": 0.50},
    "Pine": {"diameter": (30, 80), "length": (15, 35), "taper": (0.6, 1.0), "bark": (1.0, 2.0), "density": 0.40},
}


# Function to scale values using mean and std adjustment
def scale_feature(value, feature_range):
    mean = (feature_range[0] + feature_range[1]) / 2
    std_dev = (feature_range[1] - feature_range[0]) / 4  # Approximate std dev
    return mean + value * std_dev

def get_lumber_data(random_gen: random.Random, dataset: str,
                    num_points: int = 3, noise_level: float = 0.05,
                    test_size: int = 0, **kwargs):
    num_samples = num_points# +test_size
    if "class and regr" in dataset:
        # Generate classification dataset (5 classes)
        X, y = make_classification(
            n_samples=num_samples, n_features=4, n_classes=5, n_informative=3, n_redundant=1,
            n_clusters_per_class=1,
            random_state=random_gen.randint(1,10000)  # reproduce for same generator
        )
        # Normalize X values to be within [0, 1] range
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        # Assign species and scale feature values
        species = [thai_tree_species[label] for label in y]

        # Features to generate dynamically
        feature_names = ["diameter", "length", "taper", "bark"]
        df_columns = ["Log Diameter (cm)", "Log Length (m)", "Taper (cm/m)", "Bark Thickness (cm)"]

        # Scale each feature properly
        features_scaled = {
            df_columns[i]: np.array([
                scale_feature(X[j, i], scaling_factors[species[j]][feature_names[i]]) +
                              random_gen.gauss(0, noise_level * (
                                                    scaling_factors[species[j]][feature_names[i]][
                                                        1] -
                                                    scaling_factors[species[j]][feature_names[i]][
                                                        0]))

                                    for j in range(num_samples)
            ])
            for i in range(len(feature_names))
        }

        lrf_ranges = {
            "Teak": (0.65, 0.75),  # Example: 65% to 75% yield
            "Siamese Rosewood": (0.60, 0.70),
            "Bullet Wood": (0.55, 0.65),
            "Red Cedar": (0.50, 0.60),
            "Pine": (0.45, 0.55),
        }

        # Compute lumber yield using volume * density
        lumber_yield = np.array([
            (np.pi * (features_scaled["Log Diameter (cm)"][i] / 2) ** 2 *
             features_scaled["Log Length (m)"][i]) *
            scaling_factors[species[i]]["density"] *
            random_gen.uniform(*lrf_ranges[species[i]])  # Apply random LRF
            for i in range(num_samples)
        ])

        # Create DataFrame dynamically

        df_logs = pd.DataFrame(features_scaled)
        print(df_logs.shape)
        print(len(species))
        df_logs["Species"] = species
        df_logs["Lumber Yield (kg)"] = lumber_yield.round()

        rounding_rules = {
            "Log Diameter (cm)": 0,  # Round to nearest whole number
            "Log Length (m)": 1,  # Round to 1 decimal place
            "Taper (cm/m)": 1,  # Round to 1 decimal place
            "Bark Thickness (cm)": 1,  # Round to 1 decimal place
            "Lumber Yield (kg)": 0
        }
        # Apply rounding to specific columns
        for column, decimals in rounding_rules.items():
            if decimals == 0:
                # Convert to int for whole numbers
                df_logs[column] = df_logs[column].round(decimals).astype(int)
            else:
                df_logs[column] = df_logs[column].round(decimals)  # Keep decimals for others

        # Split into training and test sets
        if test_size > 0:
            # Split raw numerical data BEFORE assigning species names
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_gen.randint(1, 10000)
            )

            # Now assign species based on y_train and y_test
            species_train = [thai_tree_species[label] for label in y_train]
            species_test = [thai_tree_species[label] for label in y_test]

            # Ensure correct assignment when creating DataFrames
            df_train = pd.DataFrame(X_train,
                                    columns=["Log Diameter (cm)", "Log Length (m)", "Taper (cm/m)",
                                             "Bark Thickness (cm)"])
            df_test = pd.DataFrame(X_test,
                                   columns=["Log Diameter (cm)", "Log Length (m)", "Taper (cm/m)",
                                            "Bark Thickness (cm)"])

            df_train["Species"] = species_train
            df_test[
                "Species"] = species_test  # Only for verification, later remove Species from df_test

            # Remove species and lumber yield from test set
            df_test = df_test.drop(columns=["Species", "Lumber Yield (kg)"], errors="ignore")

            return df_train, df_test

        return df_logs, None


if __name__ == '__main__':
    df_logs, df_test = get_lumber_data(random.Random(), "Thai trees class and regr",
                                       num_points=300, test_size=0)

    print(df_test)
    print('====')
    print(df_logs)
    print(df_logs.columns)
    print(df_logs["Species"].value_counts())
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set color palette for species
    species_palette = {
        "Teak": "blue",
        "Siamese Rosewood": "red",
        "Bullet Wood": "green",
        "Red Cedar": "purple",
        "Pine": "orange",
    }
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_logs["Log Diameter (cm)"],
        y=df_logs["Log Length (m)"],
        hue=df_logs["Species"],
        palette=species_palette,
        alpha=0.7
    )
    plt.xlabel("Log Diameter (cm)")
    plt.ylabel("Log Length (m)")
    plt.title("Log Diameter vs Log Length by Species")
    plt.legend(title="Species")
    plt.grid(True)

    # Create a scatter plot for Taper vs Bark Thickness
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_logs["Taper (cm/m)"],
        y=df_logs["Bark Thickness (cm)"],
        hue=df_logs["Species"],
        palette=species_palette,
        alpha=0.7
    )
    plt.xlabel("Taper (cm/m)")
    plt.ylabel("Bark Thickness (cm)")
    plt.title("Taper vs Bark Thickness by Species")
    plt.legend(title="Species")
    plt.grid(True)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Fit LDA with 2 components
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_transformed = lda.fit_transform(df_logs[["Log Diameter (cm)", "Log Length (m)",
                                                 "Taper (cm/m)", "Bark Thickness (cm)"]],
                                        df_logs["Species"])

    # Convert LDA results to DataFrame for plotting
    df_lda = pd.DataFrame(lda_transformed, columns=["LD1", "LD2"])
    df_lda["Species"] = df_logs["Species"]

    # Plot the LDA results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_lda["LD1"], y=df_lda["LD2"], hue=df_lda["Species"], alpha=0.7)
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.title("LDA Projection of Tree Species")
    plt.legend(title="Species")
    plt.grid(True)
    plt.show()

