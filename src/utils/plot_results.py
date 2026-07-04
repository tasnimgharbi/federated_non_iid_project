import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

datasets = ["fashion", "har"]
partitions = ["iid", "label_skew", "quantity_skew"]
algorithms = ["fedavg", "fedprox", "fednova"]


def plot_metric(dataset, partition, metric, ylabel, title):
    plt.figure(figsize=(8, 5))

    for algorithm in algorithms:
        file_path = f"results/{dataset}_{partition}_{algorithm}.csv"
        df = pd.read_csv(file_path)

        plt.plot(
            df["round"],
            df[metric],
            marker="o",
            label=algorithm.upper()
        )

    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = f"plots/{dataset}_{partition}_{metric}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


for dataset in datasets:
    for partition in partitions:
        plot_metric(
            dataset,
            partition,
            "test_accuracy",
            "Global Accuracy",
            f"{dataset.upper()} - {partition} - Accuracy"
        )

        plot_metric(
            dataset,
            partition,
            "test_loss",
            "Loss",
            f"{dataset.upper()} - {partition} - Loss"
        )

        plot_metric(
            dataset,
            partition,
            "client_accuracy_std",
            "Client Accuracy STD",
            f"{dataset.upper()} - {partition} - Client Fairness"
        )
        plot_metric(
            dataset,
            partition,
            "client_accuracy_variance",
            "Client Accuracy Variance",
            f"{dataset.upper()} - {partition} - Client Variance"
        )

        plot_metric(
            dataset,
            partition,
            "communication_cost_mb",
            "Communication Cost (MB)",
            f"{dataset.upper()} - {partition} - Communication Cost"
        )

        plot_metric(
            dataset,
            partition,
            "avg_latency_ms",
            "Average Latency (ms)",
            f"{dataset.upper()} - {partition} - Network Latency"
        )