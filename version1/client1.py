from collections import OrderedDict
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

import flwr as fl
from flwr.common import Metrics, Context
from datasets.utils.logging import disable_progress_bar

DEVICE = torch.device("cpu")  # Use "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
disable_progress_bar()

NUM_CLIENTS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10


class FakeDataset():
    def __init__(self, size, transform=None, noise_level=0.1):
        self.size = size
        self.transform = transform
        self.noise_level = noise_level

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand((1, 28, 28), dtype=torch.float32)  # Add channel dimension
        label = torch.randint(0, 10, (1,)).item()            # Generate a random label from 0 to 9
        
        # Introduce noise to the label
        if torch.rand(1).item() < self.noise_level:
            label = torch.randint(0, 10, (1,)).item()
        
        if self.transform:
            image = self.transform(image)
        return image, label

def load_fake_datasets(noise_level=0.1):
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = FakeDataset(size=60000, transform=transform, noise_level=noise_level)
    test_dataset = FakeDataset(size=10000, transform=transform, noise_level=noise_level)
    
    return train_dataset, test_dataset


def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True,
                             download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    return dataset, test_dataset


def split_dataset(dataset, num_clients):
    """Splits the dataset into `num_clients` subsets."""
    client_datasets = []
    dataset_size = len(dataset)

    # Compute the size of each split
    split_size = dataset_size // num_clients
    remainder = dataset_size % num_clients

    start_idx = 0
    for i in range(num_clients):
        # Compute end index for this client
        end_idx = start_idx + split_size + (1 if i < remainder else 0)

        # Check if end_idx is within the dataset bounds
        if end_idx > dataset_size:
            end_idx = dataset_size

        client_datasets.append(torch.utils.data.Subset(
            dataset, range(start_idx, end_idx)))

        # Update start_idx for the next client
        start_idx = end_idx

    return client_datasets


def get_dataloaders(dataset, test_dataset, num_clients):
    client_datasets = split_dataset(dataset, num_clients)
    trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                    for ds in client_datasets]
    valloaders = [DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False) for _ in range(num_clients)]
    return trainloaders, valloaders


dataset, test_dataset = load_datasets()
fakedataset , test_fakedataset = load_fake_datasets(noise_level=0.2)
trainloaders, valloaders = get_dataloaders(dataset, test_dataset, NUM_CLIENTS)
faketrainloaders, fakevalloaders = get_dataloaders(fakedataset, test_fakedataset, NUM_CLIENTS)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # MNIST images have 1 channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Adjusted based on MNIST image size
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Adjusted based on MNIST image size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs: int, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(
                f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        # Compute training loss
        train_loss, train_accuracy = test(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader.dataset), {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "loss": float(loss)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data
    # if int(cid) % 3 == 0:
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # else:
        # trainloader = faketrainloaders[int(cid)]
        # valloader = fakevalloaders[int(cid)]

    return FlowerClient(net, trainloader, valloader).to_client()


# # Create FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
#     min_fit_clients=10,  # Never sample less than 10 clients for training
#     min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
#     min_available_clients=10,  # Wait until all 10 clients are available
# )

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, client_loss_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.client_loss_threshold = client_loss_threshold

    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Optional[fl.common.NDArray]:
        filtered_results = [
            (client, fit_res) for client, fit_res in results if fit_res.metrics['loss'] < self.client_loss_threshold]
        if not filtered_results:
            print(f"Round {rnd}: No clients met the loss threshold.")
            return None
        else:
            return super().aggregate_fit(rnd, filtered_results, failures)


# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

# Start simulation
strategy = CustomFedAvg(
    client_loss_threshold=0.01,
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available)
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)
