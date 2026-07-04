import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.train_eval import evaluate


def get_model_parameters(model):
    return [param.detach().cpu().clone() for param in model.parameters()]


def set_model_parameters(model, parameters):
    for param, new_param in zip(model.parameters(), parameters):
        param.data = new_param.to(param.device).clone()


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_communication_cost_mb(model, num_clients):
    """
    Communication cost per round:
    server sends model to clients + clients send model updates back.
    float32 = 4 bytes.
    """
    num_params = count_model_parameters(model)
    bytes_per_round = num_params * 4 * num_clients * 2
    return bytes_per_round / (1024 ** 2)


def simulate_network_latency(num_clients):
    """
    Potential extension:
    Simulates realistic client network latency in milliseconds.
    """
    latencies = np.random.uniform(low=20, high=300, size=num_clients)
    return float(np.mean(latencies))


def weighted_average(client_parameters, client_sizes):
    total_size = sum(client_sizes)
    averaged = []

    for params in zip(*client_parameters):
        weighted_sum = sum(
            p * size for p, size in zip(params, client_sizes)
        ) / total_size
        averaged.append(weighted_sum)

    return averaged


def train_client_fedavg(model, train_loader, device, epochs=1, lr=0.001):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps = 0

    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            steps += 1

    return model, steps


def train_client_fedprox(
    model,
    global_params,
    train_loader,
    device,
    epochs=1,
    lr=0.001,
    mu=0.01
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_params = [p.to(device) for p in global_params]
    steps = 0

    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            proximal_term = 0.0
            for param, global_param in zip(model.parameters(), global_params):
                proximal_term += torch.norm(param - global_param) ** 2

            loss = loss + (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

            steps += 1

    return model, steps


def compute_model_delta(global_params, local_params):
    return [
        local_p - global_p
        for global_p, local_p in zip(global_params, local_params)
    ]


def aggregate_fednova(global_params, client_deltas, client_sizes, client_steps):
    """
    FedNova-style normalized aggregation.

    Each client update is normalized by its number of local optimization steps.
    This reduces objective inconsistency when clients perform different amounts
    of local work.
    """
    total_size = sum(client_sizes)
    normalized_update = []

    for layer_deltas in zip(*client_deltas):
        layer_update = 0

        for delta, size, steps in zip(layer_deltas, client_sizes, client_steps):
            normalized_delta = delta / max(steps, 1)
            weight = size / total_size
            layer_update += weight * normalized_delta

        normalized_update.append(layer_update)

    avg_steps = np.average(client_steps, weights=client_sizes)

    new_global_params = [
        global_p + avg_steps * update
        for global_p, update in zip(global_params, normalized_update)
    ]

    return new_global_params


def run_federated_learning(
    model_class,
    client_datasets,
    test_loader,
    algorithm="fedavg",
    rounds=5,
    local_epochs=1,
    batch_size=64,
    lr=0.001,
    simulate_network=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = model_class().to(device)

    history = {
        "round": [],
        "test_loss": [],
        "test_accuracy": [],
        "client_accuracy_std": [],
        "client_accuracy_variance": [],
        "communication_cost_mb": [],
        "avg_latency_ms": [],
    }

    print(f"\nRunning {algorithm.upper()} on {len(client_datasets)} clients")
    print("Device:", device)

    for round_num in range(1, rounds + 1):
        global_params = get_model_parameters(global_model)

        client_parameters = []
        client_deltas = []
        client_sizes = []
        client_steps = []
        client_accuracies = []

        active_clients = 0

        for client_id, client_dataset in client_datasets.items():
            if len(client_dataset) == 0:
                continue

            active_clients += 1

            client_model = model_class().to(device)
            set_model_parameters(client_model, global_params)

            train_loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            if algorithm == "fedavg":
                client_model, steps = train_client_fedavg(
                    client_model, train_loader, device, local_epochs, lr
                )

            elif algorithm == "fedprox":
                client_model, steps = train_client_fedprox(
                    client_model, global_params, train_loader, device, local_epochs, lr
                )

            elif algorithm == "fednova":
                client_model, steps = train_client_fedavg(
                    client_model, train_loader, device, local_epochs, lr
                )

            else:
                raise ValueError("Unknown algorithm")

            local_params = get_model_parameters(client_model)

            client_parameters.append(local_params)
            client_deltas.append(compute_model_delta(global_params, local_params))
            client_sizes.append(len(client_dataset))
            client_steps.append(steps)

            _, client_acc = evaluate(client_model, train_loader, device)
            client_accuracies.append(client_acc)

        if algorithm in ["fedavg", "fedprox"]:
            new_global_params = weighted_average(client_parameters, client_sizes)

        elif algorithm == "fednova":
            new_global_params = aggregate_fednova(
                global_params,
                client_deltas,
                client_sizes,
                client_steps
            )

        set_model_parameters(global_model, new_global_params)

        test_loss, test_acc = evaluate(global_model, test_loader, device)

        client_acc_std = float(np.std(client_accuracies))
        client_acc_var = float(np.var(client_accuracies))

        communication_cost = estimate_communication_cost_mb(
            global_model,
            active_clients
        )

        avg_latency = simulate_network_latency(active_clients) if simulate_network else 0.0

        history["round"].append(round_num)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["client_accuracy_std"].append(client_acc_std)
        history["client_accuracy_variance"].append(client_acc_var)
        history["communication_cost_mb"].append(communication_cost)
        history["avg_latency_ms"].append(avg_latency)

        print(
            f"Round {round_num}/{rounds} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Loss: {test_loss:.4f} | "
            f"Client STD: {client_acc_std:.4f} | "
            f"Comm: {communication_cost:.2f} MB | "
            f"Latency: {avg_latency:.1f} ms"
        )

    return global_model, history