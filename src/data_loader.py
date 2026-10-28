import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(file_path):
    print("Loading dataset using pandas...")

    # Load the raw CICIDS2017 CSV file
    df = pd.read_csv(file_path)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Clean invalid numeric values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Convert CICIDS2017 labels into binary labels:
    # BENIGN -> 0 (safe traffic)
    # Any attack label -> 1 (malicious traffic)
    df['is_attack'] = (df['Label'].str.strip() != 'BENIGN').astype(int)

    # Remove the original text label column
    df = df.drop(columns=['Label'])

    # Keep only numeric feature columns
    df = df.select_dtypes(include=[np.number])

    # Separate features (X) and target label (y)
    X = df.drop(columns=['is_attack']).values
    y = df['is_attack'].values

    # Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features AFTER splitting to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Data loaded successfully! "
        f"Training shape: {X_train.shape}, "
        f"Testing shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test