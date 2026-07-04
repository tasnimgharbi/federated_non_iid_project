import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
from torch.utils.data import DataLoader

from src.datasets.fashion_mnist import load_fashion_mnist
from src.datasets.har import load_har
from src.datasets.partitioning import iid_partition, label_skew_partition, quantity_skew_partition
from src.models.fashion_model import FashionCNN
from src.models.har_model import HARMLP
from src.fl.simulation import run_federated_learning


os.makedirs("results", exist_ok=True)


def run_dataset_experiments(dataset_name):
    if dataset_name == "fashion":
        train_dataset, test_dataset, _, test_loader = load_fashion_mnist(batch_size=64)
        model_class = FashionCNN

    elif dataset_name == "har":
        train_dataset, test_dataset, _, test_loader = load_har(batch_size=64)
        model_class = HARMLP

    else:
        raise ValueError("Unknown dataset")

    partition_methods = {
        "iid": iid_partition,
        "label_skew": label_skew_partition,
        "quantity_skew": quantity_skew_partition,
    }

    algorithms = ["fedavg", "fedprox", "fednova"]

    all_results = []

    for partition_name, partition_func in partition_methods.items():
        clients = partition_func(train_dataset, num_clients=10)

        for algorithm in algorithms:
            _, history = run_federated_learning(
                model_class=model_class,
                client_datasets=clients,
                test_loader=test_loader,
                algorithm=algorithm,
                rounds=5,
                local_epochs=1,
                batch_size=64,
                lr=0.001,
            )

            df = pd.DataFrame(history)
            df["dataset"] = dataset_name
            df["partition"] = partition_name
            df["algorithm"] = algorithm

            all_results.append(df)

            file_name = f"results/{dataset_name}_{partition_name}_{algorithm}.csv"
            df.to_csv(file_name, index=False)
            print("Saved:", file_name)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"results/{dataset_name}_all_results.csv", index=False)


if __name__ == "__main__":
    run_dataset_experiments("fashion")
    run_dataset_experiments("har")