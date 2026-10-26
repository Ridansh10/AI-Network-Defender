import pandas as pd
import numpy as np
from sklearn .model_selection import train_test_split

def load_and_clean_data(file_path):
    print("Loading dataset using pandas...")
    # Load the raw network data CSV file
    df = pd.read_csv(file_path)

    # Cleaning the data: Drop any row that has missing values (NaN) or infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Separate features (X) from the answer key label (y)
    # here 'is_attack' will be 0 for safe traffic and 1 for malicious hacker traffic
    X = df.drop(columns=['is_attack']).values
    y = df['is_attack'].values

    # Splitting the data: 80% to train the AI, 20% held back to test the AI later
    # random_state ensures the split is the exact same every time it is run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Data loaded successfully! Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test
