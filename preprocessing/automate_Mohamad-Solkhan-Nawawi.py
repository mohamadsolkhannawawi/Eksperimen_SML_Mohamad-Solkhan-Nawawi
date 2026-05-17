"""
Automate Preprocessing - Insurance Dataset
Author: Mohamad Solkhan Nawawi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari filepath."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Duplikat dihapus: {before - after} baris | Shape: {df.shape}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding fitur kategorikal."""
    df = df.copy()

    # Binary encoding
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # One-hot encoding untuk region
    df = pd.get_dummies(df, columns=['region'], drop_first=False)

    print(f"[INFO] Encoding selesai | Kolom: {list(df.columns)}")
    return df


def handle_outliers(df: pd.DataFrame, col: str = 'bmi') -> pd.DataFrame:
    """Tangani outlier menggunakan IQR capping."""
    df = df.copy()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"[INFO] Outlier capping pada '{col}': [{lower:.2f}, {upper:.2f}]")
    return df


def split_features_target(df: pd.DataFrame, target: str = 'charges'):
    """Pisahkan fitur dan target."""
    X = df.drop(target, axis=1)
    y = df[target]
    print(f"[INFO] X shape: {X.shape} | y shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test, num_features: list):
    """Normalisasi fitur numerik menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    print(f"[INFO] Scaling selesai pada fitur: {num_features}")
    return X_train, X_test, scaler


def save_preprocessed(X_train, X_test, y_train, y_test, output_dir: str):
    """Simpan hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)

    train_data = X_train.copy()
    train_data['charges'] = y_train.values

    test_data = X_test.copy()
    test_data['charges'] = y_test.values

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"[INFO] Data tersimpan:")
    print(f"       Train: {train_path} ({train_data.shape})")
    print(f"       Test:  {test_path} ({test_data.shape})")


def run_preprocessing(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Fungsi utama pipeline preprocessing."""
    print("=" * 50)
    print("  PIPELINE PREPROCESSING - INSURANCE DATASET")
    print("=" * 50)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Hapus duplikat
    df = remove_duplicates(df)

    # Step 3: Encoding kategorikal
    df = encode_categorical(df)

    # Step 4: Tangani outlier
    df = handle_outliers(df, col='bmi')

    # Step 5: Split fitur & target
    X, y = split_features_target(df, target='charges')

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Step 7: Scaling
    num_features = ['age', 'bmi', 'children']
    X_train, X_test, _ = scale_features(X_train, X_test, num_features)

    # Step 8: Simpan
    save_preprocessed(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 50)
    print("  PREPROCESSING SELESAI!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automate Preprocessing Insurance Dataset')
    parser.add_argument('--input', type=str, default='insurance.csv', help='Path ke raw dataset')
    parser.add_argument('--output', type=str, default='insurance_preprocessing', help='Folder output')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proporsi test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')

    args = parser.parse_args()
    run_preprocessing(args.input, args.output, args.test_size, args.random_state)
