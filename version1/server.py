import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define strategy
class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()

# Define Flower server
if __name__ == "__main__":
    # Ensure that `num_rounds` and `strategy` are passed correctly
    strategy = FedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={num_rounds:3},  # Pass `num_rounds` as a keyword argument
        strategy=strategy
    )
