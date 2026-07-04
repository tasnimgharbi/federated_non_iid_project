import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.optim as optim

from src.datasets.fashion_mnist import load_fashion_mnist
from src.datasets.har import load_har
from src.models.fashion_model import FashionCNN
from src.models.har_model import HARMLP
from src.utils.train_eval import train_one_epoch, evaluate


def run_baseline(dataset_name, epochs=5, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "fashion":
        _, _, train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
        model = FashionCNN().to(device)

    elif dataset_name == "har":
        _, _, train_loader, test_loader = load_har(batch_size=batch_size)
        model = HARMLP().to(device)

    else:
        raise ValueError("dataset_name must be 'fashion' or 'har'")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nCentralized baseline for {dataset_name.upper()}")
    print("Device:", device)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    run_baseline("fashion", epochs=3)
    run_baseline("har", epochs=3)