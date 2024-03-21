import torch
import torch.nn as nn

class Model(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc = nn.Sequential(
                    nn.Linear(36192, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                    )

    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)

        x= self.conv1(x)
        x= torch.relu(x)
        x= self.conv2(x)
        x= torch.relu(x)
        x= self.conv3(x)
        x= torch.relu(x)
        x= self.pool(x)

        x = x.view(batch_size * seq_len, -1)
        # output shape of x is (batch_size, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        x = x.view(batch_size, seq_len, 2)
        return x
        