from collections import Counter
import matplotlib.pyplot as plt

from fashion_mnist import load_fashion_mnist, CLASS_NAMES


train_dataset, _, _, _ = load_fashion_mnist()

labels = [label for _, label in train_dataset]

label_counts = Counter(labels)

print("\nDataset statistics:\n")

for label, count in sorted(label_counts.items()):
    print(f"{CLASS_NAMES[label]:15} : {count}")

plt.figure(figsize=(10,5))
plt.bar(
    [CLASS_NAMES[i] for i in range(10)],
    [label_counts[i] for i in range(10)]
)

plt.xticks(rotation=45)
plt.ylabel("Number of images")
plt.title("Fashion-MNIST class distribution")
plt.tight_layout()
plt.tight_layout()
plt.savefig("plots/fashion_mnist_samples.png", dpi=300)
plt.show()