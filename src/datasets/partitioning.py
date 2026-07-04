import numpy as np
import torch
from torch.utils.data import Subset


def get_labels(dataset):
    labels = []
    for _, y in dataset:
        labels.append(int(y))
    return np.array(labels)


def iid_partition(dataset, num_clients=10):
    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, num_clients)

    return {
        client_id: Subset(dataset, split_indices[client_id].tolist())
        for client_id in range(num_clients)
    }


def label_skew_partition(dataset, num_clients=10, labels_per_client=2):
    labels = get_labels(dataset)
    unique_labels = np.unique(labels)

    client_indices = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        chosen_labels = np.random.choice(
            unique_labels,
            size=labels_per_client,
            replace=False
        )

        indices = np.where(np.isin(labels, chosen_labels))[0]
        selected = np.random.choice(
            indices,
            size=len(indices) // num_clients,
            replace=False
        )

        client_indices[client_id].extend(selected.tolist())

    return {
        client_id: Subset(dataset, client_indices[client_id])
        for client_id in range(num_clients)
    }


def quantity_skew_partition(dataset, num_clients=10, alpha=0.5):
    proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
    proportions = proportions / proportions.sum()

    indices = np.random.permutation(len(dataset))
    split_points = (np.cumsum(proportions) * len(dataset)).astype(int)[:-1]
    split_indices = np.split(indices, split_points)

    return {
        client_id: Subset(dataset, split_indices[client_id].tolist())
        for client_id in range(num_clients)
    }