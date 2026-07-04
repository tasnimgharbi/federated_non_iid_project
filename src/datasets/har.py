import os
import zipfile
import urllib.request
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def download_and_extract_har(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)

    outer_zip = os.path.join(data_dir, "har.zip")
    extract_path = os.path.join(data_dir, "HAR")

    final_path = os.path.join(extract_path, "UCI HAR Dataset")

    if os.path.exists(final_path):
        print("HAR already extracted.")
        return final_path

    if not os.path.exists(outer_zip):
        print("Downloading HAR dataset...")
        urllib.request.urlretrieve(HAR_URL, outer_zip)

    os.makedirs(extract_path, exist_ok=True)

    print("Extracting outer zip...")
    with zipfile.ZipFile(outer_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    inner_zip = None
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".zip"):
                inner_zip = os.path.join(root, file)

    if inner_zip is not None:
        print("Extracting inner zip...")
        with zipfile.ZipFile(inner_zip, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    if not os.path.exists(final_path):
        raise FileNotFoundError("Could not find UCI HAR Dataset after extraction.")

    return final_path


def load_har(data_dir="data", batch_size=64):
    base = download_and_extract_har(data_dir)

    X_train = pd.read_csv(
        os.path.join(base, "train", "X_train.txt"),
        sep=r"\s+",
        header=None
    ).values

    y_train = pd.read_csv(
        os.path.join(base, "train", "y_train.txt"),
        header=None
    ).values.ravel() - 1

    X_test = pd.read_csv(
        os.path.join(base, "test", "X_test.txt"),
        sep=r"\s+",
        header=None
    ).values

    y_test = pd.read_csv(
        os.path.join(base, "test", "y_test.txt"),
        header=None
    ).values.ravel() - 1

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader = load_har()

    print("HAR loaded successfully")
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    X, y = next(iter(train_loader))
    print("Feature batch shape:", X.shape)
    print("Label batch shape:", y.shape)