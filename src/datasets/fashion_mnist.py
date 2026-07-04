import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_fashion_mnist(data_dir="data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, test_dataset, train_loader, test_loader


def show_sample_images(train_loader):
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(8, 8))

    for i in range(9):
        image = images[i].squeeze()
        label = labels[i].item()

        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(CLASS_NAMES[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader = load_fashion_mnist()

    print("Fashion-MNIST loaded successfully")
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    show_sample_images(train_loader)