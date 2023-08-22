import torch
from torch import nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.flatten(x, start_dim=1)
        padding = 27446 - x.size(1)
        if padding > 0:
            x = F.pad(x, (0, padding), "constant", 0.0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)