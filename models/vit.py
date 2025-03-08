import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VIT(nn.Module):
    """ Simple CNN model used to demonstrate embeddings can be used as input"""

    def __init__(self, input_channels, shape=(64*64)):
        """ Input_channels: could be image channels (RGB) or emhbedding size
            Shape: shape of input image
        """
        super(VIT, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((np.prod(shape)*8)//16, 16)
        self.fc2 = nn.Linear(16, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return without activation, is done in loss function
        return x