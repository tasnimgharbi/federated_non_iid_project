import os
import pandas as pd

os.makedirs("results/tables", exist_ok=True)

datasets = ["fashion", "har"]
partitions = ["iid", "label_skew", "quantity_skew"]
algorithms = ["fedavg", "fedprox", "fednova"]

dataset_names = {
    "fashion": "Fashion-MNIST",
    "har": "HAR"
}

partition_names = {
    "iid": "IID",
    "label_skew": "Label-Skew",
    "quantity_skew": "Quantity-Skew"
}

algorithm_names = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fednova": "FedNova"
}

rows = []

for dataset in datasets:
    for partition in partitions:
        table_rows = []

        for algorithm in algorithms:
            file_path = f"results/{dataset}_{partition}_{algorithm}.csv"
            df = pd.read_csv(file_path)
            final = df.iloc[-1]

            row = {
                "Algorithm": algorithm_names[algorithm],
                "Final Accuracy (%)": round(final["test_accuracy"] * 100, 2),
                "Final Loss": round(final["test_loss"], 4),
                "Client STD": round(final["client_accuracy_std"], 4),
                "Client Variance": round(final["client_accuracy_variance"], 4),
                "Communication Cost (MB)": round(final["communication_cost_mb"], 2),
                "Avg. Latency (ms)": round(final["avg_latency_ms"], 2),
            }

            table_rows.append(row)

            rows.append({
                "Dataset": dataset_names[dataset],
                "Partition": partition_names[partition],
                **row
            })

        table = pd.DataFrame(table_rows)
        csv_path = f"results/tables/{dataset}_{partition}_table.csv"
        md_path = f"results/tables/{dataset}_{partition}_table.md"

        table.to_csv(csv_path, index=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                f"### {dataset_names[dataset]} — {partition_names[partition]} Results\n\n"
            )
            f.write(table.to_markdown(index=False))

summary = pd.DataFrame(rows)
summary.to_csv("results/tables/overall_summary.csv", index=False)

with open("results/tables/overall_summary.md", "w", encoding="utf-8") as f:
    f.write("### Overall Experimental Summary\n\n")
    f.write(summary.to_markdown(index=False))

print("Generated Word-ready tables in results/tables/")
print(summary)