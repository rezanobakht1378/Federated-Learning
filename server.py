#!/usr/bin/env python
# coding: utf-8

# In[4]:


# pip install -q flwr[simulation] torch torchvision
# pip install flwr_datasets[vision]


# In[2]:


# pip install matplotlib


# In[3]:


from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import flwr as fl

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[5]:


def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# In[6]:


from datasets import Dataset
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar

# Let's set a simulation involving a total of 100 clients
NUM_CLIENTS = 10

# Download MNIST dataset and partition the "train" partition (so one can be assigned to each client)
mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
# Let's keep the test set as is, and use it to evaluate the global model on the server
centralized_testset = mnist_fds.load_split("test")


# In[7]:


from torchvision.transforms import ToTensor, Normalize, Compose


def apply_transforms(batch):
    """Get transformation for MNIST dataset"""

    # transformation to convert images to tensors and apply normalization
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


# In[ ]:

from flwr.common import NDArrays, Scalar
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, client_loss_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.client_loss_threshold = client_loss_threshold

    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException])-> Optional[fl.common.Weights]:
        filtered_results = [
            (client, fit_res) for client, fit_res in results if fit_res.metrics['loss'] < self.client_loss_threshold]
        if not filtered_results:
            print(f"Round {rnd}: No clients met the loss threshold.")
            return None
        return super().aggregate_fit(rnd, filtered_results, failures)
                      


# In[9]:


client_resources = {"num_cpus": 1, "num_gpus": 0.0}
disable_progress_bar()


# In[ ]:


fl.server.start_server(
  server_address="0.0.0.0:8080",
  sterategy=CustomFedAvg(),
  config=fl.server.ServerConfig(num_rounds=3),
)

