import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MLP, self).__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        return x