"""yolov8-kitty-fedavg: A Flower / YOLOv8 Federated app."""

import os
import shutil
from typing import Tuple


# -------------------------------------------------------------
# DATASET HANDLING FOR YOLOv8 + KITTY
# -------------------------------------------------------------
# We assume the KITTY dataset is in YOLO format:
#
# kitty/
#   images/train/
#   images/val/
#   labels/train/
#   labels/val/
#   kitty.yaml
#
# Federated version:
#   kitty_client_0.yaml
#   kitty_client_1.yaml
#   ...
#
# -------------------------------------------------------------


def create_partition_yaml(
    partition_id: int,
    num_partitions: int,
    base_dir: str = "./kitty",
) -> str:
    """
    Create a KITTY YAML file for each client partition.
    Each client sees a different subset of train/val data.
    """

    yaml_path = f"./kitty_client_{partition_id}.yaml"

    if os.path.exists(yaml_path):
        return yaml_path

    # Each client receives a split:
    #   train_{id}/
    #   val_{id}/

    # So we expect:
    # kitty/
    #   images/train_0/
    #   images/train_1/
    #   ...
    #   labels/train_0/
    #   labels/train_1/
    #
    # (Same for val directories)

    yaml_text = f"""
path: {base_dir}

train: images/train_{partition_id}
val: images/val_{partition_id}

names:
  0: car
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    return yaml_path


# -------------------------------------------------------------
# LOAD DATA FOR CLIENT
# -------------------------------------------------------------

def load_data(partition_id: int, num_partitions: int) -> Tuple[str, str]:
    """
    Returns the train/val YAML path for this client.
    This replaces all DataLoader, FederatedDataset, and transforms.
    YOLOv8 expects a YAML file, not Dataloaders.
    """

    yaml_path = create_partition_yaml(partition_id, num_partitions)

    # return dummy values to satisfy the client
    # actual training uses the YAML path inside model.train()
    return yaml_path, yaml_path


# -------------------------------------------------------------
# TRAIN / TEST PLACEHOLDERS (NOT USED IN YOLOv8)
# -------------------------------------------------------------

def train(*args, **kwargs):
    """YOLO training is handled directly in ClientApp via model.train()"""
    return None


def test(*args, **kwargs):
    """YOLO evaluation is handled directly in ClientApp via model.val()"""
    return None


# """yolov8-kitty-fedavg: A Flower / PyTorch app."""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Normalize, ToTensor


# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# fds = None  # Cache FederatedDataset

# pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#     return batch


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition Kitty data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="kitty.yaml",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     # Construct dataloaders
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader


# def train(net, trainloader, epochs, lr, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     net.eval()
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy
