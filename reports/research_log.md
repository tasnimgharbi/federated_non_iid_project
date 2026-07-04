# Federated Learning on non-IID Data

## Research Log

---

## Session 1

### Objective

Set up the development environment and prepare the first dataset.

### Tasks Completed

- Installed Python 3.13
- Created a virtual environment
- Installed all required libraries:
  - PyTorch
  - Torchvision
  - Flower
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - tqdm
- Created the project structure.
- Downloaded the Fashion-MNIST dataset.
- Loaded the training and test datasets.
- Visualized sample images.
- Explored the class distribution.

### Observations

Fashion-MNIST contains:

- 60,000 training images
- 10,000 testing images
- 10 clothing categories
- Balanced class distribution (6000 images per class)

The dataset is therefore suitable for creating artificial IID and non-IID client partitions later in the project.

### Next Step

Load and preprocess the HAR dataset.