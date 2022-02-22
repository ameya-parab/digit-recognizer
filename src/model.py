import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)

        self.dropout1 = nn.Dropout2d(p=0.20)
        self.dropout2 = nn.Dropout2d(p=0.35)

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)


    def forward(self, data):

        data = self.conv1(data)
        data = F.leaky_relu(data)
        data = F.max_pool2d(data, 2)
        data = self.dropout1(data)

        data = self.conv2(data)
        data = F.leaky_relu(data)
        data = F.max_pool2d(data, 2)
        data = self.dropout2(data)

        data = torch.flatten(data, start_dim=1)

        data = self.fc1(data)
        data = F.leaky_relu(data)
        data = self.fc2(data)
        data = F.leaky_relu(data)
        output = self.fc3(data)

        return output
