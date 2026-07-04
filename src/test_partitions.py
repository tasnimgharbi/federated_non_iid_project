import sys
import os
sys.path.append(os.path.abspath("."))

from src.datasets.fashion_mnist import load_fashion_mnist
from src.datasets.partitioning import iid_partition, label_skew_partition, quantity_skew_partition


train_dataset, _, _, _ = load_fashion_mnist()

for name, func in [
    ("IID", iid_partition),
    ("Label-skew", label_skew_partition),
    ("Quantity-skew", quantity_skew_partition),
]:
    clients = func(train_dataset, num_clients=10)
    print(f"\n{name} partition:")
    for client_id, client_data in clients.items():
        print(f"Client {client_id}: {len(client_data)} samples")